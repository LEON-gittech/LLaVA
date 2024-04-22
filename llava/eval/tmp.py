import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

device = "cuda:0"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    print(model)
    print(image_processor)
    # # Process the fixed example images once
    fixed_images = []
    fixed_image_sizes = []
    fixed_examples = [
        {"image": "862902619928506372.jpg"},
        {"image": "827619548610310148.jpg"}
    ]
    
    for example in fixed_examples:
        image_path = os.path.join(args.image_folder, example["image"])
        image = Image.open(image_path).convert('RGB')
        # 保存图像尺寸
        fixed_image_sizes.append(image.size)  # image.size 返回一个元组 (width, height)
        
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        fixed_images.append(image_tensor)



    print("question file", args.question_file)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        original_text = line["text"]

        # Build the qs with few-shot example description
        qs = (
        "You are a sarcasm expert, you can accurately identify whether the multimodal "
        "message is sarcasm through the image and text pair. "
        "As a sarcasm expert, you are provided with two reference examples to help identify sarcasm in multimodal messages. "
        "Example 1 (Sarcastic): 'i am guessing # netflix no longer lets you grab screens of movies . that & the new rating system is so awesome . ' associated with its image. "
        "Now, i show you the picture of Example 1 (Sarcastic):"
        "Example 2 (Not Sarcastic): 'follow wishful widows on facebook ... # divorce needs # inspiration and ...' associated with its image. "
        "Now, i show you the picture of Example 2 (Not Sarcastic):"
        f"Now, given an image and text pair, the text is \"{original_text}\". May I ask whether the image and text express "
        "the meaning of sarcasm to the message. Review the image and determine whether the combined image and text express sarcasm. "
        "You should choose the answer from 'sarcastic' or 'not sarcastic'."
        )
        cur_prompt = qs

        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs = (
        "You are a sarcasm expert, you can accurately identify whether the multimodal "
        "message is sarcasm through the image and text pair. "
        "As a sarcasm expert, you are provided with two reference examples to help identify sarcasm in multimodal messages. "
        "Example 1 (Sarcastic): 'i am guessing # netflix no longer lets you grab screens of movies . that & the new rating system is so awesome . ' associated with its image. "
        "Now, i show you the picture of Example 1 (Sarcastic):"
        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"  # 图片插入于第一个例子末尾
        "Example 2 (Not Sarcastic): 'follow wishful widows on facebook ... # divorce needs # inspiration and ...' associated with its image. "
        "Now, i show you the picture of Example 2 (Not Sarcastic):"
        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"  # 图片插入于第二个例子末尾
        f"Now, given an image and text pair, the text is \"{original_text}\". May I ask whether the image and text express "
        "the meaning of sarcasm to the message. Review the image and determine whether the combined image and text express sarcasm. "
        "You should choose the answer from 'sarcastic' or 'not sarcastic'."
        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"  # 图片插入于最后问题末尾
        )

        # # 初始化原始qs字符串，不包含图像标记
        # qs = (
        # "You are a sarcasm expert, you can accurately identify whether the multimodal "
        # "message is sarcasm through the image and text pair. "
        # "As a sarcasm expert, you are provided with two reference examples to help identify sarcasm in multimodal messages. "
        # "Example 1 (Sarcastic): 'i am guessing # netflix no longer lets you grab screens of movies . that & the new rating system is so awesome . ' associated with its image. "
        # "Now, I show you the picture of Example 1 (Sarcastic):"
        # )

        # # 插入第一张图片
        # if model.config.mm_use_im_start_end:
        #     qs += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        # else:
        #     qs += DEFAULT_IMAGE_TOKEN + "\n"

        # qs += (
        # "Example 2 (Not Sarcastic): 'follow wishful widows on facebook ... # divorce needs # inspiration and ...' associated with its image. "
        # "Now, I show you the picture of Example 2 (Not Sarcastic):"
        # )

        # # 插入第二张图片
        # if model.config.mm_use_im_start_end:
        #     qs += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        # else:
        #     qs += DEFAULT_IMAGE_TOKEN + "\n"

        # qs += (
        # f"Now, given an image and text pair, the text is \"{original_text}\". May I ask whether the image and text express "
        # "the meaning of sarcasm to the message. Review the image and determine whether the combined image and text express sarcasm. "
        # "You should choose the answer from 'sarcastic' or 'not sarcastic'."
        # )

        # # 插入第三张图片
        # if model.config.mm_use_im_start_end:
        #     qs += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        # else:
        #     qs += DEFAULT_IMAGE_TOKEN + "\n"


        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        qs_image = Image.open(os.path.join(args.image_folder, image_file)+".jpg").convert('RGB')
        question_image_tensor = process_images([qs_image], image_processor, model.config)[0]

        # Combine the fixed example images with the current question image
        fixed_images = [x.unsqueeze(0) for x in fixed_images]
        question_image_tensor = question_image_tensor.unsqueeze(0)
        all_images_tensor = torch.cat(fixed_images + [question_image_tensor], dim=0)

        # 同样保存问题图像的尺寸
        question_image_size = qs_image.size  # (width, height)

        # 构建图像尺寸列表，用于模型输入
        image_sizes = fixed_image_sizes + [question_image_size]

        print("images:",all_images_tensor.half().to(device).shape)
        print("image_sizes:",image_sizes)
        #print("input_ids:",input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                #images=image_tensor.unsqueeze(0).half().cuda(),
                images=all_images_tensor.half().to(device),  # Assuming the model and tensors are compatible
                #image_sizes=[image.size],
                image_sizes=image_sizes,  # 确保这里传递的是一个列表，包含每张图片的尺寸元组
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/opt/tiger/LLaVA/llava/eval/dataset_image")
    parser.add_argument("--question-file", type=str, default="/opt/tiger/LLaVA/llava/eval/test_hlj.jsonl")
    parser.add_argument("--answers-file", type=str, default="/opt/tiger/LLaVA/llava/eval/answer2.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)