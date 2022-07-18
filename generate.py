from utils.dataset import get_datasets
import torch


if __name__ == "__main__":

    train_loader, test_loader = get_datasets(
        type="mqm9", batch_size=1000, shuffle=True, num_workers=4
    )

    batch_count = 100
    pos_list = []
    mask_list = []
    validity_list = []

    for idx, batch in enumerate(train_loader):
        if idx >= batch_count:
            break
        
        print(f"Processing batch {idx}")
        pos = batch.pos
        mask = batch.mask

        # Add to the valid list

        valid_validity = torch.ones(pos.shape[0])

        pos_list.append(pos)
        mask_list.append(mask)
        validity_list.append(valid_validity)

        # Generate invalid data

        # generate random number

        invalid_pos = torch.randn_like(pos)
        invalid_validity = torch.zeros(pos.shape[0])

        pos_list.append(invalid_pos[:50])
        mask_list.append(mask[:50])
        validity_list.append(invalid_validity[:50])

        # Add normal noise to the pos

        invalid_addition = torch.randn_like(pos)
        invalid_pos = pos + invalid_addition * mask.unsqueeze(2)
        
        pos_list.append(invalid_pos[:50])
        mask_list.append(mask[:50])
        validity_list.append(invalid_validity[:50])

        # Create discontinuity

        invalid_addition = torch.randint_like(pos, low=0, high=2)
        invalid_pos = pos + invalid_addition * mask.unsqueeze(2)

        pos_list.append(invalid_pos[:100])
        mask_list.append(mask[:100])
        validity_list.append(invalid_validity[:100])

        # Perturn the position of an atom within a molecule

        invalid_pos = pos.clone()
        for degree in range(8):
            random_ratio = torch.rand(pos.shape[0])
            random_int = torch.floor(random_ratio * mask.sum(dim=-1)).long()

            invalid_pos[torch.arange(pos.shape[0]), random_int] += torch.randn(pos.shape[0], 3) * 2

            pos_list.append(invalid_pos[:100].clone())
            mask_list.append(mask[:100])
            validity_list.append(invalid_validity[:100])



    
    final_pos = torch.cat(pos_list, dim=0)
    final_mask = torch.cat(mask_list, dim=0)
    final_validity = torch.cat(validity_list, dim=0)

    
    rand_perm = torch.randperm(final_pos.shape[0])

    final_pos = final_pos[rand_perm]
    final_mask = final_mask[rand_perm]
    final_validity = final_validity[rand_perm]

    print(final_pos[:20], final_mask[:20], final_validity[:20])
    print(final_pos.shape)


    data_objects = {
        "pos": final_pos,
        "mask": final_mask,
        "validity": final_validity
    }

    torch.save(data_objects, "data_v3.pt")