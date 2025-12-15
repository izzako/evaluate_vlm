import asyncio
import argparse
import base64
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from PIL import Image
from datasets import load_dataset


class VLMResponse(BaseModel):
    """Structured output from VLM"""
    description: str = Field(..., description="The generated text/sentence description")


@dataclass
class BenchmarkResult:
    """Results from benchmark run"""
    image_path: str
    text_id: int
    chunk_id: int
    response: str
    latency: float
    success: bool
    timestamp: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VLMBenchmark:
    """Benchmark suite for Vision Language Models via OpenAI-compatible API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        output_file: str = "results.jsonl",
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.output_file = Path(output_file)
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or clear the output file
        if self.output_file.exists():
            print(f"Warning: Output file {self.output_file} already exists. Appending to it.")
        else:
            self.output_file.touch()
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def encode_pil_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def get_image_url(self, image_path: str) -> str:
        """Get data URL for image file"""
        ext = Path(image_path).suffix.lower()
        mime_type = f"image/{ext[1:]}" if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'] else "image/jpeg"
        
        base64_img = self.encode_image(image_path)
        return f"data:{mime_type};base64,{base64_img}"
    
    def get_pil_image_url(self, image: Image.Image) -> str:
        """Get data URL for PIL Image"""
        base64_img = self.encode_pil_image(image)
        return f"data:image/png;base64,{base64_img}"
    
    def process_image_input(self, image_input: Any) -> tuple[Image.Image, str]:
        """
        Process various image input formats and return PIL Image and path string
        
        Args:
            image_input: Can be:
                - str: file path
                - PIL Image
                - list: pixel values
                - dict with 'pixel_values' key
        
        Returns:
            tuple: (PIL Image, image_path_string)
        """
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input), image_input
        elif isinstance(image_input, Image.Image):
            # Already PIL Image
            return image_input, "pil_image"
        elif isinstance(image_input, list):
            # list (pixel values)
            img = Image.fromarray(np.array(image_input, dtype=np.uint8))
            return img, "pixel_values"
        elif isinstance(image_input, dict) and 'pixel_values' in image_input:
            # Dict with pixel_values key
            img = Image.fromarray(np.array(image_input['pixel_values'], dtype=np.uint8))
            return img, "pixel_values"
        elif hasattr(image_input, '__array__'):
            # Array-like object
            arr = np.array(image_input, dtype=np.uint8)
            img = Image.fromarray(arr)
            return img, "pixel_values"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def save_result(self, result: BenchmarkResult):
        """Save a single result to JSONL file (live update)"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False)
            f.write('\n')
            f.flush()  # Force write to disk immediately
    
    async def inference_single(
        self,
        image_input: Any,  # Can be path or PIL Image
        prompt: str,
        text_id: int,
        chunk_id: int,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_structured_output: bool = True,
        fewshot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> BenchmarkResult:
        """Run inference on a single image"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Process image input to get PIL Image
        try:
            pil_image, base_path = self.process_image_input(image_input)
            image_path = f"{base_path}_{text_id}_{chunk_id}"
            image_url = self.get_pil_image_url(pil_image)
        except Exception as e:
            # If image processing fails, return error immediately
            return BenchmarkResult(
                image_path=str(image_input)[:100],
                text_id=text_id,
                chunk_id=chunk_id,
                response="",
                latency=time.time() - start_time,
                success=False,
                timestamp=timestamp,
                error=f"Image processing error: {str(e)}",
                metadata=metadata
            )
        
        try:
            # Prepare messages with optional few-shot examples
            messages = []
            
            # Add few-shot examples if provided
            if fewshot_examples:
                for example in fewshot_examples:
                    # Process example image
                    example_pil, _ = self.process_image_input(example['image'])
                    example_image_url = self.get_pil_image_url(example_pil)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": example_image_url}}
                        ]
                    })
                    # Assistant response for the example
                    messages.append({
                        "role": "assistant",
                        "content": example['response']
                    })
            
            # Add the actual query image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
            
            # Make API call with structured output
            if use_structured_output:
                try:
                    completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "vlm_response",
                                "strict": True,
                                "schema": VLMResponse.model_json_schema()
                            }
                        }
                    )
                    
                    # Parse structured output
                    content = completion.choices[0].message.content
                    
                    # Try to parse as JSON
                    try:
                        parsed = VLMResponse.model_validate_json(content)
                        response_text = parsed.description
                    except Exception as parse_error:
                        # If JSON parsing fails, try to extract description manually
                        print(f"Warning: JSON parsing failed for {image_path}, attempting manual extraction")
                        print(f"Parse error: {parse_error}")
                        print(f"Content: {content[:500]}...")
                        
                        # Try to extract description from malformed JSON
                        try:
                            import re
                            match = re.search(r'"description"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', content)
                            if match:
                                response_text = match.group(1).replace('\\"', '"')
                            else:
                                # Try without quotes
                                match = re.search(r'"description"\s*:\s*([^,}\]]+)', content)
                                if match:
                                    response_text = match.group(1).strip()
                                else:
                                    # Fallback: use the raw content
                                    response_text = content
                        except:
                            response_text = content
                            
                except Exception as api_error:
                    # If structured output fails, retry without it
                    print(f"Warning: Structured output failed for {image_path}, retrying without structure")
                    print(f"Error: {api_error}")
                    
                    completion = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    response_text = completion.choices[0].message.content
            else:
                # Regular completion without structured output
                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response_text = completion.choices[0].message.content
            
            latency = time.time() - start_time
            
            benchmark_result = BenchmarkResult(
                image_path=image_path,
                text_id=text_id,
                chunk_id=chunk_id,
                response=response_text,
                latency=latency,
                success=True,
                timestamp=timestamp,
                metadata=metadata
            )
            
        except Exception as e:
            latency = time.time() - start_time
            benchmark_result = BenchmarkResult(
                image_path=image_path,
                text_id=text_id,
                chunk_id=chunk_id,
                response="",
                latency=latency,
                success=False,
                timestamp=timestamp,
                error=str(e),
                metadata=metadata
            )
        
        # Save result immediately
        self.save_result(benchmark_result)
        return benchmark_result
        
    async def inference_batch(
        self,
        examples: List[Dict[str, Any]],
        prompt: str,
        batch_size: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_structured_output: bool = True,
        fewshot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> List[BenchmarkResult]:
        """Run batch inference on multiple examples with concurrency control"""
        results = []
        total_batches = (len(examples) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            tasks = [
                self.inference_single(
                    image_input=ex['image'],
                    prompt=prompt,
                    text_id=ex['text_id'],
                    chunk_id=ex['chunk_id'],
                    metadata=ex.get('metadata'),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_structured_output=use_structured_output,
                    fewshot_examples=fewshot_examples
                )
                for ex in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            batch_num = i // batch_size + 1
            successful = sum(1 for r in batch_results if r.success)
            print(f"Processed batch {batch_num}/{total_batches} - {successful}/{len(batch_results)} successful")
        
        return results
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print benchmark summary statistics"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total examples: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            latencies = [r.latency for r in successful]
            print(f"\nLatency Statistics:")
            print(f"  Mean: {sum(latencies)/len(latencies):.3f}s")
            print(f"  Min: {min(latencies):.3f}s")
            print(f"  Max: {max(latencies):.3f}s")
            print(f"  Total time: {sum(latencies):.3f}s")
            print(f"  Throughput: {len(successful)/sum(latencies):.3f} examples/s")
        
        if failed:
            print(f"\nFailed examples:")
            for r in failed[:10]:  # Show first 10 failures
                print(f"  {r.image_path}: {r.error}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        
        print(f"\nResults saved to: {self.output_file}")
        print("="*60)
    
    async def close(self):
        """Close the OpenAI client"""
        await self.client.close()


async def run_benchmark(args):
    """Main benchmark runner"""
    # Determine output folder
    if args.debug:
        output_folder = Path(args.output_folder) / "debug"
    else:
        output_folder = Path(args.output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"results_{args.split}_{timestamp}.jsonl"
    
    # Initialize benchmark
    benchmark = VLMBenchmark(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        output_file=str(output_file),
        timeout=args.timeout,
        max_retries=args.max_retries
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}, split: {args.split}")
    dataset = load_dataset(args.dataset_path, split=args.split)
    
    # Load few-shot examples if provided
    fewshot_examples = None
    if args.fewshot_file:
        print(f"Loading few-shot examples from {args.fewshot_file}")
        fewshot_examples = []
        with open(args.fewshot_file, 'r', encoding='utf-8') as f:
            fewshot_data = json.load(f)
            for item in fewshot_data:
                # Support multiple image formats
                if 'image_path' in item:
                    # Load from file path
                    fewshot_examples.append({
                        'image': item['image_path'],
                        'response': item['response']
                    })
                elif 'pixel_values' in item:
                    # Load from pixel values
                    fewshot_examples.append({
                        'image': item['pixel_values'],
                        'response': item['response']
                    })
                elif 'image' in item:
                    # Generic image field
                    fewshot_examples.append({
                        'image': item['image'],
                        'response': item['response']
                    })
        print(f"Loaded {len(fewshot_examples)} few-shot examples")
    
    # Debug mode: limit to 10 examples
    if args.debug:
        print("Debug mode enabled: processing only 10 examples")
        dataset = dataset.select(range(min(10, len(dataset))))
    
    print(f"Total examples to process: {len(dataset)}")
    
    # Prepare examples
    examples = []
    for idx, item in enumerate(dataset):
        # Handle different image field names and formats
        image_data = None
        if 'image' in item:
            image_data = item['image']
        elif 'pixel_values' in item:
            image_data = item['pixel_values']
        elif 'img' in item:
            image_data = item['img']
        else:
            print(f"Warning: No image field found in item {idx}, skipping...")
            continue
            
        examples.append({
            'image': image_data,
            'text_id': item['text_id'],
            'chunk_id':item['chunk_id'],
            'metadata': {
                'source_language': args.source_language,
                'label_text': item['text']
                # Add any other relevant metadata from the dataset
            }
        })
    
    # Use prompt from arguments
    prompt = args.prompt
    
    try:
        # Run batch inference
        print(f"Starting batch inference with batch_size={args.batch_size}...")
        if fewshot_examples:
            print(f"Using {len(fewshot_examples)}-shot learning")
        start_time = time.time()
        
        results = await benchmark.inference_batch(
            examples=examples,
            prompt=prompt,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            use_structured_output=not args.no_structured_output,
            fewshot_examples=fewshot_examples
        )
        
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.2f}s")
        
        # Print summary
        benchmark.print_summary(results)
        
    finally:
        await benchmark.close()


def main():
    parser = argparse.ArgumentParser(description='VLM Benchmark Tool')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to the dataset (Hugging Face dataset name or local path)')
    parser.add_argument('--split', type=str, required=True, 
                       help='Dataset split (e.g., train, test, validation)')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name to use for inference (Hugging Face path)')
    parser.add_argument('--source_language', type=str, required=True, 
                       help='Source language/script for transliteration and translation')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt to use for all images')
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='Folder to save output results')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Number of examples per batch (default: 32)')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1',
                       help='Base URL for the OpenAI-compatible API (default: http://localhost:8000/v1)')
    parser.add_argument('--api_key', type=str, default='',
                       help='API key for authentication (default: empty)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling (default: 0.7)')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--timeout', type=float, default=120.0,
                       help='Request timeout in seconds (default: 120.0)')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retries for failed requests (default: 3)')
    parser.add_argument('--no_structured_output', action='store_true',
                       help='Disable structured output (use plain text response)')
    parser.add_argument('--fewshot_file', type=str, default=None,
                       help='Path to JSON file containing few-shot examples')
    parser.add_argument('--debug', action='store_true', 
                       help='Run in debug mode with only 10 examples and save outputs in debug folder')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("BENCHMARK CONFIGURATION")
    print("="*60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model_name}")
    print(f"Base URL: {args.base_url}")
    print(f"Source Language: {args.source_language}")
    print(f"Prompt: {args.prompt}")
    print(f"Output Folder: {args.output_folder}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Structured Output: {not args.no_structured_output}")
    print(f"Few-shot File: {args.fewshot_file if args.fewshot_file else 'None'}")
    print(f"Debug Mode: {args.debug}")
    print("="*60 + "\n")
    
    # Run benchmark
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()