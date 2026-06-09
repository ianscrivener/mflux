# Branch Read Me

***Note:** This branch code builds on and requires **`scriv-mixed-quant-inference-flux2`**.* 

**Model Formats**
 - Gen AI image models are typically released on HuggingFace in pytorch/HuggingFace safetensors format
 - MLX, and hence MFlux, uses a slightly different safetensors format - so conversion is required

**Quantization**
 - quantizing is a standard form of lossy compression used to reduce the size (and hence required memory and on disk size) of models
 - quantizing works by reducing the precision of values in the vector representations of the models
 - bf16 will support 65,536 distinct values - while q4 will only support 256 distinct values
 - quantization can either be;
   1. in training quantization - Quantization‑aware training (QAT)
   2. post training - PTQ: post-training quantization
 - Most people associate quantization with PTQ, as QAT is a more recent innovation
 - QAT delivers substantially more quality than the equivalent PTQ - eg Nvidia's nvfp4 QAT achieve results approaching PTQ q8
   
**NVFP4**
 - NVFP4 is an Nvidia quantization format that is baked into the hardware of the recent Blackwell/RTX-50 series GPUs.
 - Nvidia uses the term QAD (Quantization-Aware Distillation) to denote QAT
 - QAT NVFP4 has the same number of bits as other 4‑bit formats, but delivers a higher‑quality signal within that 4‑bit budget.
 - Apple hardware does not currently support NVFP4 
 - However, recent versions of MLX have software convertor for NVFP4 - allowing NVFP4 models to be run
 - anecdotally, the QAT NVFP4 'signal quality' can be thought of as being somewhere in the range of PTQ Q6/Q8. 

**Current State**
 - Mflux offers `mflux-save` to convert models from a pytorch/HuggingFace safetensors format to MLX/Mflux safetensor format
 - there are 3 model types: **text_encoder**, **transformer**, **vae**
 - the default no quantising
 - quantizing options are 3, 4, 5, 6 and 8 bits. 
 - when the quantizing option is selected all three models (text_encoder, transformer & vae), are saved with that same quantizing. 

**Where 'signal quality' & quantization matters more** 
 - typically it is considered that;
   1. **VAE** quality matters most. So usually V's are kept at BF16, sometimes FP32.
      VAEs are much smaller (168Mb) models and so have less disk size and memory implications as well.
   2. **transformer** quality matters, but not as much as VAEs. Models are large (8Gb for q8).
   3. **text_encoder** quality matters least. Models are also large (8Gb for q8).
       
**Improvements in this branch**
1. Add `mflux-save-advanced`to convert models with separately quantizations for vae, transformer and text_encoder
     `mflux-save-advanced`
     `  --bf16_model `
     `  --quantize_vae               # 3, 4, 5, 6, 8, bf16`
     `  --quantize_transformer       # 3, 4, 5, 6, 8, bf16`
     `  --quantize_text_encoder      # 3, 4, 5, 6, 8, bf16`
     
2. Support conversion from *nvfp4 pytorch safetensors format* to *MLX safetensors* format for transformers only
	   `mflux-save-advanced`
	   `  --nvfp4_model `
	    ` --quantize_transformer      # nvfp4 only`
   

