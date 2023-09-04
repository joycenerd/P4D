from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
from typing import Any, Mapping
from statistics import mean
import numpy as np
import argparse
import random
import torch
import copy
import json
import os


# def set_random_seed(seed=0):
#     torch.manual_seed(seed + 0)
#     torch.cuda.manual_seed(seed + 1)
#     torch.cuda.manual_seed_all(seed + 2)
#     np.random.seed(seed + 3)
#     torch.cuda.manual_seed_all(seed + 4)
#     random.seed(seed + 5)

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def initialize_prompt(tokenizer, token_embedding, args, device):
    prompt_len = args.prompt_len

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    # padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    padded_template_text = template_text.format(" ".join(["<|startoftext|>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)
    dummy_ids = dummy_ids[1:-1]
    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    # no need to pad the first and last tokens because SD tokenizer has already done padding
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False

    return prompt_embeds, dummy_embeds, dummy_ids


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))
    # print(f'texts: {texts}')
    return texts


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)
        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        if torch.__version__ >= "1.13.0":
            ori_batch = torch.concatenate(ori_batch).to(device)
        else:
            ori_batch = torch.cat(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        if torch.__version__ >= "1.13.0":
            gen_batch = torch.concatenate(gen_batch).to(device)
        else:
            gen_batch = torch.cat(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()


# def dummy(images, **kwargs):
#     return images, False


def optimize(clip_model, clip_preprocess, img_preprocess, pipe, generator, erase_pipe, erase_generator, target_prompt, negative_prompt, target_imgs, guidance, safe_config, img_save_dir, args):

    # tokenizer, token embedding, intialize prompt
    tokenizer = pipe.tokenizer
    token_embedding = pipe.text_encoder.text_model.embeddings.token_embedding

    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, args.device)
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=args.lr, weight_decay=args.weight_decay)

    # safety guidance
    enable_safety_guidance = False
    if safe_config is not None and safe_config["sld_guidance_scale"] >= 1:
        enable_safety_guidance = True
        safety_momentum = None

    with torch.no_grad():
        curr_images = [img_preprocess(i).unsqueeze(0) for i in target_imgs]
        if torch.__version__ >= "1.13.0":
            curr_images = torch.concatenate(curr_images).to(args.device)
        else:
            curr_images = torch.cat(curr_images).to(args.device)
        all_latents = pipe.vae.encode(curr_images.to(args.weight_dtype)).latent_dist.sample()
        all_latents = all_latents * 0.18215

    best_loss = -999
    eval_loss = -99999
    best_text = ""

    for step in range(args.iter):
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding)
        tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True
    
            
        # padding and repeat
        padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds[:, 1:args.prompt_len+1] = tmp_embeds
        padded_embeds = padded_embeds.repeat(args.batch_size, 1, 1)
        padded_dummy_ids = dummy_ids.repeat(args.batch_size, 1)
        
        # randomly sample images and get features
        if args.batch_size is None:
            latents = all_latents
        else:
            perm = torch.randperm(len(all_latents))
            idx = perm[:args.batch_size]
            latents = all_latents[idx]
        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape, generator=generator, device=args.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
        timesteps = timesteps.long()
            
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # get text embeddings
        input_text_embeddings = pipe._new_encode_prompt(target_prompt, args.num_images_per_prompt, do_classifier_free_guidance=0, negative_prompt=None)
        padded_text_embeddings = pipe._get_text_embedding_with_embeddings(padded_dummy_ids, padded_embeds)
        
        if args.filter:
            if enable_safety_guidance:
                padded_text_embeddings = erase_pipe._expand_safe_text_embeddings(padded_text_embeddings.cuda(args.device_2), args.num_images_per_prompt)
            elif negative_prompt is not None:
                padded_text_embeddings = erase_pipe._expand_uncond_text_embeddings(padded_text_embeddings.cuda(args.device_2), negative_prompt, args.num_images_per_prompt)

        # UNet forward
        target_model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=input_text_embeddings).sample
        if args.filter and (enable_safety_guidance or negative_prompt is not None):
            latent_model_input = torch.cat([noisy_latents]*2).cuda(args.device_2)
            latent_model_input = erase_pipe.scheduler.scale_model_input(latent_model_input, timesteps.cuda(args.device_2))
        else:
            latent_model_input = noisy_latents.cuda(args.device_2)
        padded_model_pred = erase_pipe.unet(latent_model_input, timesteps.cuda(args.device_2), encoder_hidden_states=padded_text_embeddings.cuda(args.device_2)).sample
        
        if args.filter:
            # Perform guidance if enable safety guidance
            if enable_safety_guidance:
                padded_model_pred_text, noise_pred_safety_concept = padded_model_pred.chunk(2)
                # noise_pred_uncond, padded_model_pred_text, noise_pred_safety_concept = padded_model_pred.chunk(3)
                noise_guidance = padded_model_pred_text
                # noise_guidance = (padded_model_pred_text - noise_pred_uncond)
            
                # Perform SLD guidance
                if safety_momentum is None:
                    safety_momentum = torch.zeros_like(noise_guidance)
                tmp_momentum = safety_momentum.clone().detach()

                # Equation 6
                scale = torch.clamp(
                    torch.abs((padded_model_pred_text - noise_pred_safety_concept)) * safe_config["sld_guidance_scale"], max=1.)

                # Equation 6
                safety_concept_scale = torch.where(
                    (padded_model_pred_text - noise_pred_safety_concept) >= safe_config["sld_threshold"],
                    torch.zeros_like(scale), scale)

                # Equation 4
                noise_guidance_safety = torch.mul(noise_pred_safety_concept, safety_concept_scale)
                # noise_guidance_safety = torch.mul(
                #             (noise_pred_safety_concept - noise_pred_uncond), safety_concept_scale)

                # Equation 7
                # noise_guidance_safety = noise_guidance_safety + safe_config["sld_momentum_scale"] * safety_momentum
                noise_guidance_safety = noise_guidance_safety + safe_config["sld_momentum_scale"] * tmp_momentum

                # Equation 8
                # safety_momentum = safe_config["sld_mom_beta"] * safety_momentum + (1 - safe_config["sld_mom_beta"]) * noise_guidance_safety
                tmp_momentum = safe_config["sld_mom_beta"] * tmp_momentum + (1 - safe_config["sld_mom_beta"]) * noise_guidance_safety

                if step >= safe_config["sld_warmup_steps"]: # Warmup
                    # Equation 3
                    noise_guidance = noise_guidance - noise_guidance_safety
            
                padded_model_pred = guidance * noise_guidance
            
            # perform guidance if negative prompt is not None
            elif negative_prompt is not None:
                noise_pred_uncond, padded_model_pred_text = padded_model_pred.chunk(2)
                padded_model_pred = noise_pred_uncond + guidance * (padded_model_pred_text - noise_pred_uncond)

        padded_model_pred = padded_model_pred.cuda(args.device)
        loss = torch.nn.functional.mse_loss(padded_model_pred.float(), target_model_pred.float(), reduction="mean")
        
        ### update prompt
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
        if args.filter and enable_safety_guidance:
            safety_momentum = tmp_momentum.detach()
            del tmp_momentum
        input_optimizer.step()
        input_optimizer.zero_grad()    
        
        curr_lr = input_optimizer.param_groups[0]["lr"]

        # clear cuda cache
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
        
        ### eval
        if step % args.eval_step == 0:
            decoded_text = decode_ids(nn_indices, tokenizer)[0]
                
            with torch.no_grad():
                pred_imgs = erase_pipe(
                    decoded_text,
                    negative_prompt=args.negative_prompts,
                    num_images_per_prompt=args.num_images_per_prompt,
                    guidance_scale=guidance,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    generator=erase_generator,
                    **(safe_config or {})

                ).images
                eval_loss = measure_similarity(target_imgs, pred_imgs, clip_model, clip_preprocess, args.device)

            if best_loss < eval_loss:
                best_loss = eval_loss
                best_text = decoded_text
                # best_embeds = copy.deepcopy(prompt_embeds.detach())
        
            if step % args.print_step == 0:
                print(f"step: {step}, lr: {curr_lr}, cosim: {eval_loss:.3f}, best_cosim: {best_loss:.3f}, best prompt: {best_text}")
                save_path = f'{img_save_dir}/{str(step)}.png'
                pred_imgs[0].save(save_path)
    
    print()
    print(f"Best shot: cosine similarity: {best_loss:.3f}")
    print(f"text: {best_text}")
    return best_text, best_loss


if __name__ == "__main__":
    args = argparse.Namespace()
    args.__dict__.update(read_json("sample_config.json"))
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    optimize(args.target_prompts, args)
