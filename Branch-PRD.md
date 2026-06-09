# Branch PRD


### Task 1
Add `mflux-save-advanced`to convert models with separately quantizations for vae, transformer and text_encoder
     `mflux-save-advanced`
	`  --bf16_model                 # REQUIRED a HuggingFace repo org/model, or a local path
     `  --quantize_vae               # OPTIONAL 3, 4, 5, 6, 8, bf16`
     `  --quantize_transformer       # OPTIONAL 3, 4, 5, 6, 8, bf16`
     `  --quantize_text_encoder      # OPTIONAL 3, 4, 5, 6, 8, bf16`
     `  --path PATH                  # REQUIRED Local path for saving a model to disk.`

Mflux already supports quantizing models into a variety of bit formats. However, currently `mflux-save`  converts the text_encoder, transformer and VAE to the specified quantized action bits.   

This update allows the quantization for each of the text encoder, transformer and Vi to be set separately. 


**Goals**
1. converting bf16 pytorch safetensors format > bf16 MLX safetensors format
     ie `--quantize_XXX bf16`` (slightly inaccurate argument naming)
2. converting bf16 pytorch safetensors format > bf16 MLX safetensors format & quantizing to 3, 4, 5, 6 or 8bit
	  ie `--quantize_XXX 8`
   
   
**Non-goals**
3. converting from MLX safetensors format
4. converting from fp32 pytorch safetensors format
5. converting from any quantized pytorch safetensors format

**Rules**
 - Only convert and optionally quantize the VAE if `--quantize_vae` is present
 - Only convert and optionally quantize the VAE if `--quantize_transformer` is present
 - Only convert and optionally quantize the VAE if `--quantize_text_encoder` is present
 - End immediately with a error message if the source model is not bf16 and pytorch safetensors format
 - Ask for user confirmation on overwrite 

---

### Task 2
Support conversion from *nvfp4 pytorch safetensors format* to *MLX safetensors* format for transformers only
	   `mflux-save-advanced`
	   `  --nvfp4_model `

**Goals**
1. convert nvfp4 pytorch safetensors format > nvfp4 MLX safetensors format
2. Ask for user confirmation on overwrite 

**Rules**
 - convert from any format other than nvfp4 pytorch safetensors format 
 - convert to any format other than nvfp4 MLX safetensors format 
 - convert any model other than type transformer

**Rules**
 - End immediately with a error message if the source model is not nvfp4 and pytorch safetensors format 
 - Ask for user confirmation on overwrite 