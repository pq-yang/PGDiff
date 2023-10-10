import argparse
import os
import cv2
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    SUPPORTED_TASKS,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_restorer,
    add_dict_to_argparser,
    args_to_dict,
    create_arcface_embedding,
    avg_grayscale,
    adaptive_instance_normalization,
)


def main():

    def partial_guidance(x, t, y=None, pred_xstart=None, target=None, ref=None, mask=None, task="restoration", scale=0, N=1, s_start=1, s_end=0.7): 
        assert y is not None
        with th.enable_grad():
            pred_xstart_in = pred_xstart.detach().requires_grad_(True)

            total_loss = 0
            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            # property: lightness & color stats
            if task == 'colorization':
                mse_loss = F.mse_loss(avg_grayscale(y), avg_grayscale(pred_xstart_in), reduction='sum') * args.lightness_weight # 1
                print(f'loss (lightness): {mse_loss};', end=' ')
                total_loss = total_loss + mse_loss

                pred_xstart_adain = adaptive_instance_normalization(pred_xstart_in, None).clamp(-1,1)
                adain_loss = F.mse_loss(pred_xstart_in, pred_xstart_adain, reduction='sum') * args.color_weight # 0.05
                print(f'loss (color): {adain_loss};', end=' ')
                total_loss = total_loss + adain_loss
            
            # property: unmasked region
            if task == 'inpainting':
                mse_loss = F.mse_loss(y[mask==0], pred_xstart_in[mask==0], reduction='sum') * args.unmasked_weight # 1
                print(f'loss (unmasked): {mse_loss};', end=' ')
                total_loss = total_loss + mse_loss

            # property: smooth semantics
            if 'restoration' in task:
                if target == None:
                    fake_g_output = restorer(x, y_t=y, t=t).clamp(-1,1)
                    fake_g_output = fake_g_output.detach().requires_grad_(True).cuda()
                else:
                    fake_g_output = target.detach().requires_grad_(True).cuda()
                mse_loss = F.mse_loss(fake_g_output, pred_xstart_in, reduction='sum') * args.ss_weight # 1
                print(f'loss (smooth semantics): {mse_loss};', end=' ')
                total_loss = total_loss + mse_loss

            # + property: identity reference
            if task == 'ref_restoration':
                emd_x0 = embedding(F.interpolate(pred_xstart_in, (112,112), mode='bilinear', antialias=True))
                emd_ref = embedding(F.interpolate(ref, (112,112), mode='bilinear', antialias=True))
                # Choice 1: MSE loss (default)
                ref_loss = F.mse_loss(emd_x0, emd_ref, reduction='sum') * args.ref_weight # 25
                # Choice 2: Cosine loss
                # cos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
                # ref_loss = -cos(emd_x0, emd_ref) * args.ref_weight # 1e4
                print(f'loss (identity): {ref_loss};', end=' ')
                total_loss = total_loss + ref_loss

            # composite tasks
            if task == "old_photo_restoration":
                total_loss = 0
                pred_xstart_in = pred_xstart.detach().requires_grad_(True)
                fake_g_output = fake_g_output.detach().requires_grad_(True)
                mse_loss = F.mse_loss(avg_grayscale(fake_g_output)[mask==0], avg_grayscale(pred_xstart_in)[mask==0], reduction='sum') * args.op_lightness_weight # 1
                print(f'loss (lightness): {mse_loss};', end=' ')
                total_loss = total_loss + mse_loss

                pred_xstart_adain = adaptive_instance_normalization(pred_xstart_in, None).clamp(-1,1)
                adain_loss = F.mse_loss(pred_xstart_in, pred_xstart_adain, reduction='sum') * args.op_color_weight # 0.5
                print(f'loss (color): {adain_loss};', end=' ')
                total_loss = total_loss + adain_loss

            if t.cpu().numpy()[0] > 0:
                print(end='\r')
            else:
                print('\n')
            gradient = th.autograd.grad(total_loss, pred_xstart_in)[0]
            if args.task == "inpainting" or args.task == "old_photo_restoration":
                gradient[mask>0] = 0
        if 'restoration' in task:
            return gradient, fake_g_output.detach()
        else:
            return gradient, None
    

    def model_fn(x, t, y=None, target=None, ref=None, mask=None, task=None, scale=0, N=1, s_start=1, s_end=0.7):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = f'{args.out_dir}/s{args.guidance_scale}-seed{args.seed}'
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict({key[7:]:value for key, value in state_dict.items()})
    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())
    model.eval()

    if 'restoration' in args.task:
        logger.log("Loading restorer for smooth semantics prediction...")
        restorer = create_restorer()
        restorer.load_state_dict(
            dist_util.load_state_dict(args.restorer_path, map_location="cpu")['state_dict'], strict=False
        )
        restorer.to(dist_util.dev())
        restorer.eval()
    
    if args.task == 'ref_restoration':
        logger.log("Loading embedding for identity feature extraction...")
        embedding = create_arcface_embedding()
        embedding.load_state_dict(
            dist_util.load_state_dict("models/ms1mv3_arcface_r50_fp16.pth")
        )
        embedding.to(dist_util.dev())
        embedding.eval()
    
    assert args.task in SUPPORTED_TASKS, "Task not supported!"

    print("=================== Summary (Sampling) ===================")

    print(f'Task: {args.task}; Guidance scale: {args.guidance_scale}')
    if args.N > 1:
        print(f'From {args.s_start}T to {args.s_end}T, {args.N} gradient steps are taken at each time step.')

    if args.task == 'colorization':
        print(f'Apply partial guidance on lightness (w={args.lightness_weight}).') 
        print(f'Apply partial guidance on color stats (w={args.color_weight}).')
    elif args.task == 'inpainting':
        print(f'Apply partial guidance on unmasked regions (w={args.unmasked_weight}).')
    elif args.task == 'restoration':
        print(f'Apply partial guidance on smooth semantics (w={args.ss_weight}).')
    elif args.task == 'ref_restoration':
        print(f'Apply partial guidance on smooth semantics (w={args.ss_weight}).')
        print(f'Apply partial guidance on identity ref (w={args.ref_weight}).')
    elif args.task == 'old_photo_restoration':
        print(f'Apply partial guidance on old photo lightness (w={args.op_lightness_weight}).')
        print(f'Apply partial guidance on old photo color stats (w={args.op_color_weight}).')
    
    print("==========================================================")
    
    

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    
    all_images = []
    lr_folder = args.in_dir
    lr_images = sorted(os.listdir(lr_folder))

    if args.task == 'ref_restoration':
        ref_folder = args.ref_dir
        assert ref_folder is not None, "Please input reference folder!"
        assert len(os.listdir(ref_folder)) == len(os.listdir(lr_folder)), "The number of images in the input folder and reference folder should match!"
    if args.task == 'inpainting' or args.task == 'old_photo_restoration':
        mask_folder = args.mask_dir
        if mask_folder is None:
            print(f'No mask is inputted!')

    logger.log("Sampling...")

    for img_name in lr_images:

        model_kwargs = {}
        model_kwargs["task"] = args.task
        model_kwargs["target"] = None
        model_kwargs["scale"] = args.guidance_scale
        model_kwargs["N"] = args.N
        model_kwargs["s_start"] = int(args.s_start * args.diffusion_steps)
        model_kwargs["s_end"] = int(args.s_end * args.diffusion_steps)
        y0 = cv2.resize(cv2.imread(osp.join(lr_folder, img_name)), (512,512)).astype(np.float32)[:, :, [2, 1, 0]]/ 127.5 - 1
        model_kwargs["y"] = th.tensor(y0).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [-1,1]
        if args.task == 'ref_restoration':
            ref_img = cv2.resize(cv2.imread(osp.join(ref_folder, img_name)), (512,512)).astype(np.float32)[:, :, [2, 1, 0]]/ 127.5 - 1
            model_kwargs["ref"] = th.tensor(ref_img).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [-1,1]
        elif args.task == 'inpainting' or args.task == 'old_photo_restoration':
            try:
                mask_img = cv2.resize(cv2.imread(osp.join(mask_folder, img_name)), (512,512)).astype(np.float32)/ 255.
            except:
                print('Warning: Will treat as if there are no missing pixels!')
                mask_img = np.zeros((512, 512, 3)).astype(np.float32)/ 255.
            model_kwargs["mask"] = th.tensor(mask_img).permute(2,0,1).unsqueeze(0).cuda() # (B,C,H,W), [0,1]
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=partial_guidance,
            device=dist_util.dev(),
            seed=seed
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(f'{out_dir}/{img_name}', all_images[-1][0][...,[2,1,0]])

    dist.barrier()
    logger.log("Sampling complete!")

def create_argparser():
    defaults = dict(
        seed=1234,
        task='restoration',
        in_dir='testdata/cropped_faces',
        out_dir='results/blind_restoration',
        ref_dir=None,
        mask_dir=None,
        lightness_weight=1.0,
        color_weight=0.05,
        unmasked_weight=1.0,
        ss_weight=1.0,
        ref_weight=25.0,
        op_lightness_weight=1.0,
        op_color_weight=0.5,
        N=1,                        # number of gradient steps at each time t
        s_start=1.0,                # range for multiple gradient steps (S_{start} = s_start * T)
        s_end=0.7,                  # range for multiple gradient steps (S_{end} = s_end * T)
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="models/iddpm_ffhq512_ema500000.pth",
        restorer_path="models/restorer/rrdb_iter_100000.pth",
        guidance_scale=0.1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()