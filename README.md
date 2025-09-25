# KnowMT-Bench

Benchmarking Knowledge-Intensive Long Form Question Answering for LLMs in Multi-Turn Dialogues.

## Get Start

```bash
# Create environment
conda create -n knowmt python=3.10
conda activate knowmt

# Install dependencies
pip install -r requirements.txt
```

## Usage


### Basic Evaluation
```bash
python eval.py --model qwen2.5-7b \
                --data_path data/KnowMT_QA.json \
                --output_path results/ \
```

### Arguments

Required:
- `--model`: Model name
- `--data_path`: Path to dataset
- `--output_path`: Output directory

Optional:
- `--mode {single,multi}`: Evaluation mode (default: multi)
- `--limit`: Max samples to evaluate
- `--resume_from_stage {1,2,3}`: Resume from stage

## Three-Stage Pipeline

1. **Generate**: Target model generates responses
2. **Decompose**: Extract factual statements
3. **Evaluate**: Assess factuality and hallucinations

Results saved in unified JSON format with metrics like `F1_fact`, `recall_fact`, etc.

## License

MIT License