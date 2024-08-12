import torch
from pathlib import Path
import warnings
from transformers import DistilBertTokenizer, DistilBertModel


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'------------------- device : {str(device)} ------------------------')

    tokenizer = DistilBertTokenizer.from_pretrained(
        'monologg/distilkobert',
        clean_up_tokenization_spaces=False
    )

    model = (DistilBertModel.from_pretrained("monologg/distilkobert", torchscript=True)
             .to(device))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        example_text = "Replace me by any text you'd like."
        encoded_input = tokenizer(example_text, return_tensors='pt').to(device)
        traced_model = torch.jit.trace(model, [encoded_input['input_ids'], encoded_input['attention_mask']])

    # save
    model_path = Path("./model_repository/distilbert/1/model.pt")
    model_path.parent.mkdir(exist_ok=True)
    tokenizer_path = Path("./model_repository/tokenizer/resource")
    tokenizer_path.parent.mkdir(exist_ok=True)

    torch.jit.save(traced_model, str(model_path))
    print(f"save model -> {model_path.absolute()}")
    tokenizer.save_pretrained(str(tokenizer_path))
    print(f"save tokenizer -> {tokenizer_path.absolute()}")
