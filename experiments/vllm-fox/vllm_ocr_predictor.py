#!/usr/bin/env python3
"""
vLLM-based inference module for OCR Vision Language Models.
Optimized for batch processing with Qwen2.5-VL models.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
try:
    # DeepSeek-OCR specific logits processor (available in vLLM nightly)
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
except Exception:
    NGramPerReqLogitsProcessor = None

logger = logging.getLogger(__name__)


class OCRVLMPredictor:
    """
    A class for batch inference with Vision Language Models using vLLM for OCR tasks.
    
    This class provides efficient batch prediction capabilities for VLMs,
    specifically optimized for OCR tasks with the Qwen2.5-VL series of models.
    
    Attributes:
        model_name: The name or path of the model to use.
        llm: The vLLM LLM instance.
        sampling_params: Default sampling parameters for generation.
        system_prompt: System prompt for OCR tasks.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_model_len: int = 8192,
        limit_mm_per_prompt: dict[str, int] | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        system_prompt_path: str = "system_prompt.txt",
        skip_mm_profiling: bool = False,
        # OpenAI API (vLLM serve) path for models like nanonets/Nanonets-OCR-s
        use_openai_api: Optional[bool] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OCR VLM predictor with vLLM.
        
        Args:
            model_name: The name or path of the model to load.
            max_model_len: Maximum sequence length for the model.
            limit_mm_per_prompt: Dictionary specifying limits for multimodal inputs.
            tensor_parallel_size: Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0).
            system_prompt_path: Path to system prompt file.
            **kwargs: Additional arguments to pass to vLLM's LLM constructor.
        """
        self.model_name = model_name
        self.use_openai_api = use_openai_api if use_openai_api is not None else False
        self.api_base_url = api_base_url or os.environ.get("VLLM_OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_OPENAI_API_KEY", "EMPTY")
        
        # Set default multimodal limits if not provided
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 1}  # OCR typically uses single images
        
        # Load system prompt
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                self.system_prompt = f.read().strip()
            logger.info(f"Loaded system prompt from {system_prompt_path}")
        else:
            self.system_prompt = "You are an expert OCR assistant. Please read and analyze the provided images accurately."
            logger.warning(f"System prompt file not found at {system_prompt_path}, using default")
        
        logger.info(f"Initializing vLLM with model: {model_name}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"Multimodal limits: {limit_mm_per_prompt}")
        logger.info(f"Skip MM profiling: {skip_mm_profiling}")
        # If loading from a local checkpoint that lacks HF processor files, populate from base instruct.
        try:
            if os.path.isdir(model_name):
                preproc_path = os.path.join(model_name, "preprocessor_config.json")
                if not os.path.exists(preproc_path):
                    logger.warning(
                        "Image processor config missing in checkpoint. "
                        "Falling back to base instruct processor and saving into the local directory."
                    )
                    from transformers import AutoProcessor
                    base_processor_src = "Qwen/Qwen2.5-VL-7B-Instruct"
                    processor = AutoProcessor.from_pretrained(base_processor_src, trust_remote_code=True)
                    # Save the processor files into the local checkpoint directory so vLLM can find them
                    processor.save_pretrained(model_name)
                    logger.info(f"Saved processor files from {base_processor_src} to {model_name}")
        except Exception as e:
            logger.warning(f"Failed to populate processor files: {e}")
        # If using OpenAI API path (e.g., vLLM serve), initialize client instead of local LLM
        if self.use_openai_api:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
                logger.info(f"Using OpenAI-compatible API at {self.api_base_url}")
            except Exception:
                raise RuntimeError("openai>=1.0 is required for OpenAI API mode but not installed")
            self.llm = None
        else:
            # Initialize vLLM locally with a single engine creation
            is_deepseek = "deepseek-ai/DeepSeek-OCR" in model_name or model_name.endswith("DeepSeek-OCR")
            llm_kwargs = dict(
                model=model_name,
                max_model_len=max_model_len,
                limit_mm_per_prompt=limit_mm_per_prompt,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                skip_mm_profiling=skip_mm_profiling,
            )
            llm_kwargs.update(kwargs)

            # DeepSeek-OCR recommendations: disable prefix caching and mm cache; add logits processor
            if is_deepseek:
                llm_kwargs.setdefault("enable_prefix_caching", False)
                llm_kwargs.setdefault("mm_processor_cache_gb", 0)
                if NGramPerReqLogitsProcessor is not None:
                    lp = llm_kwargs.get("logits_processors") or []
                    # Avoid duplicate addition
                    if NGramPerReqLogitsProcessor not in [getattr(p, "__class__", type(p)) for p in lp]:
                        lp.append(NGramPerReqLogitsProcessor)
                    llm_kwargs["logits_processors"] = lp
                else:
                    logger.warning("DeepSeek-OCR logits processor not available. Ensure vLLM nightly is installed.")

            # Initialize vLLM once
            self.llm = LLM(**llm_kwargs)
        
        # Initialize tokenizer for token-level metrics
        try:
            # trust_remote_code is needed for some specialized models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            self.tokenizer = None
        
        # Default sampling parameters for OCR (typically want deterministic output)
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding for consistency
            max_tokens=2048,  # OCR output can be long
            top_p=1.0,
        )
        
        logger.info("vLLM initialization complete")
    
    def format_prompt(
        self,
        text_prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format prompt according to the target model.
        - DeepSeek-OCR prefers plain prompts: "<image>\n{text_prompt}"
        - Qwen family (Qwen2/Qwen2.5/Qwen3) should use the tokenizer's chat template
          when available to ensure correct special tokens for images and roles.
        """
        is_deepseek = "deepseek-ai/DeepSeek-OCR" in self.model_name or self.model_name.endswith("DeepSeek-OCR")
        # For OpenAI API mode (e.g., Nanonets), we return plain text prompt; image is attached separately
        if self.use_openai_api:
            return text_prompt
        if is_deepseek:
            # Ensure image token prefix
            prefix = "<image>\n"
            if text_prompt.startswith("<image>"):
                return text_prompt
            return f"{prefix}{text_prompt}"

        # Prefer tokenizer chat template for Qwen models if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Construct messages with an image placeholder understood by Qwen tokenizers
                sys_msg = system_prompt or self.system_prompt
                messages = []
                if sys_msg:
                    messages.append({"role": "system", "content": sys_msg})
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]})
                # Generate the formatted prompt without tokenizing
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.warning(f"Falling back to manual Qwen prompt formatting: {e}")

        # Fallback to manual Qwen-style formatting if chat template unavailable
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            return (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def prepare_batch_prompts(
        self,
        samples: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare batch of prompts for vLLM inference.
        
        Args:
            samples: List of sample dictionaries with image paths and text prompts.
            system_prompt: Optional system prompt to override default.
            
        Returns:
            List of formatted prompt dictionaries for vLLM.
        """
        vllm_prompts = []
        
        for sample in samples:
            # Extract components
            image_path = sample['image_path']
            text_prompt = sample['text_prompt']
            
            # Format the text prompt according to model
            # In OpenAI API mode, we prepare message dicts with base64 images
            if self.use_openai_api:
                # Load and encode image
                try:
                    img = Image.open(image_path)
                    import io
                    import base64
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception:
                    img_b64 = None
                user_text = self.format_prompt(text_prompt=text_prompt, system_prompt=system_prompt)
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"} if img_b64 else {"url": ""},
                        },
                        {"type": "text", "text": user_text},
                    ],
                }]
                vllm_prompts.append({
                    "messages": messages,
                    "image_path": image_path,
                    "raw_prompt": user_text,
                })
            else:
                # Format the text prompt according to model for local vLLM
                formatted_text = self.format_prompt(
                    text_prompt=text_prompt,
                    system_prompt=system_prompt
                )
                
                # Load image
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    logger.error(f"Failed to load image {image_path}: {e}")
                    continue
                
                # Create prompt dict for vLLM
                vllm_prompt = {
                    "prompt": formatted_text,
                    "multi_modal_data": {"image": image}
                }
                vllm_prompts.append(vllm_prompt)
        
        return vllm_prompts
    
    def predict_batch(
        self,
        samples: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        **sampling_kwargs
    ) -> List[str]:
        """
        Perform batch prediction on a list of samples.
        
        Args:
            samples: List of sample dictionaries with 'image_path' and 'text_prompt'.
            temperature: Sampling temperature (0.0 for greedy).
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            system_prompt: Optional system prompt to override default.
            **sampling_kwargs: Additional sampling parameters.
            
        Returns:
            List of generated text responses.
        """
        # Create sampling parameters (local vLLM only)
        is_deepseek = "deepseek-ai/DeepSeek-OCR" in self.model_name or self.model_name.endswith("DeepSeek-OCR")
        if self.use_openai_api:
            sampling_params = None
        elif is_deepseek:
            # Include DeepSeek-specific extras
            extra_args = sampling_kwargs.pop("extra_args", {})
            # Sensible defaults from DeepSeek docs
            extra_args.setdefault("ngram_size", 30)
            extra_args.setdefault("window_size", 90)
            extra_args.setdefault("whitelist_token_ids", {128821, 128822})
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                skip_special_tokens=False,
                extra_args=extra_args,
                **sampling_kwargs
            )
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **sampling_kwargs
            )
        
        logger.info(f"Running batch inference on {len(samples)} samples")
        
        # Prepare prompts for vLLM
        vllm_prompts = self.prepare_batch_prompts(samples, system_prompt)
        
        if not vllm_prompts:
            logger.error("No valid prompts to process")
            return []
        
        results = []
        if self.use_openai_api:
            # Call OpenAI-compatible API per sample
            for item in vllm_prompts:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=item.get("messages") or [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": ""}},
                                    {"type": "text", "text": item.get("prompt", "")},
                                ],
                            }
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                except Exception as e:
                    logger.error(f"OpenAI API call failed: {e}")
                    text = ""
                results.append(text)
        else:
            # Run inference (vLLM handles batching automatically)
            outputs = self.llm.generate(
                vllm_prompts,
                sampling_params=sampling_params
            )
            
            # Extract generated text
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                results.append(generated_text)
        
        logger.info(f"Batch inference complete. Generated {len(results)} responses")
        return results
    
    def predict_single(
        self,
        image_path: str,
        text_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **sampling_kwargs
    ) -> str:
        """
        Convenience method for single prediction.
        
        Args:
            image_path: Path to the image file.
            text_prompt: The text instruction/question.
            system_prompt: Optional system prompt to override default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **sampling_kwargs: Additional sampling parameters.
            
        Returns:
            Generated text response.
        """
        sample = {
            'image_path': image_path,
            'text_prompt': text_prompt
        }
        
        results = self.predict_batch(
            [sample],
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **sampling_kwargs
        )
        return results[0] if results else ""
    
    def evaluate_dataset(
        self,
        gt_data: List[Dict[str, Any]],
        image_dir: str,
        extract_prompt_fn,
        batch_size: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a complete dataset with batch processing.
        
        Args:
            gt_data: Ground truth data loaded from JSON.
            image_dir: Directory containing images.
            extract_prompt_fn: Function to extract prompt from ground truth entry.
            batch_size: Optional batch size (vLLM handles this automatically if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt to override default.
            progress_callback: Optional callback for progress reporting.
            
        Returns:
            List of evaluation results.
        """
        logger.info(f"Starting evaluation on {len(gt_data)} samples")
        
        # Prepare all samples for batch processing
        samples = []
        valid_indices = []
        
        for i, ann in enumerate(gt_data):
            try:
                # Get image path
                image_file = ann["image"]
                image_file_path = os.path.join(image_dir, image_file)
                
                if not os.path.exists(image_file_path):
                    logger.warning(f"Image not found: {image_file_path}")
                    continue
                
                # Load image to get size information
                try:
                    image = Image.open(image_file_path)
                    image_size = image.size  # (width, height)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_file_path}: {e}")
                    continue
                
                # Extract user prompt using provided function (with image size if supported)
                user_prompt = extract_prompt_fn(ann, image_size=image_size)
                
                samples.append({
                    'image_path': image_file_path,
                    'text_prompt': user_prompt,
                    'original_index': i,
                    'annotation': ann
                })
                valid_indices.append(i)
                
            except Exception as e:
                logger.error(f"Error preparing sample {i}: {str(e)}")
                continue
        
        logger.info(f"Prepared {len(samples)} valid samples for processing")
        
        # Process in batches if batch_size is specified, otherwise let vLLM handle it
        if batch_size:
            all_results = []
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
                
                batch_results = self.predict_batch(
                    batch_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )
                all_results.extend(batch_results)
                
                if progress_callback:
                    progress_callback(i + len(batch_samples), len(samples))
        else:
            # Let vLLM handle batching automatically
            all_results = self.predict_batch(
                samples,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
        
        # Format output
        output_list = []
        for i, (sample, result) in enumerate(zip(samples, all_results)):
            ann = sample['annotation']
            
            output_json = {
                'image': ann["image"],
                'question': sample['text_prompt'],
                'label': ann["conversations"][1]["value"],
                'answer': result,
                'sample_id': sample['original_index']
            }
            output_list.append(output_json)
        
        logger.info(f"Evaluation complete. Processed {len(output_list)} samples")
        return output_list
