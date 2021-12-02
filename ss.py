def test(args, model, test_loader, device, tokenizer):
    total=0
    right=0
    for test_batch in tqdm(test_loader, disable=args.local_rank not in [-1, 0]):
        #query_id, doc_id, label= test_batch[''], test_batch['doc_id'], test_batch['label']
        with torch.no_grad():
            if args.original_t5:
                output_sequences = model.module.generate(
                    input_ids=test_batch['input_ids'].to(device),
                    attention_mask=test_batch['attention_mask'].to(device),
                    do_sample=False, # disable sampling to test if batching affects output
                )
                batch_result= tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
               # print(batch_result)
                # print(batch_result)
                true_result = test_batch["raw_label"]
                total += len(true_result)
                for br, tr in zip(batch_result, true_result):
                    if br == tr:
                        right += 1
            else:
                batch_score = model(
                        input_ids=test_batch['input_ids'].to(device), 
                        attention_mask=test_batch['attention_mask'].to(device), 
                        decoder_input_ids=test_batch['decoder_input_ids'].to(device),
                        )
                predict=torch.argmax(batch_score,dim=1)
                label=test_batch['label'].to(device)
                total+=len(label)
                right+=torch.eq(predict,label).sum()
    # return int(total),int(right.detach().cpu())
    return total, right