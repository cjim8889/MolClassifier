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

        pos_list.append(invalid_pos[:200])
        mask_list.append(mask[:200])
        validity_list.append(invalid_validity[:200])

        # Add normal noise to the pos

        invalid_addition = torch.randn_like(pos) * 2
        invalid_pos = pos + invalid_addition * mask.unsqueeze(2)
        
        pos_list.append(invalid_pos[:500])
        mask_list.append(mask[:500])
        validity_list.append(invalid_validity[:500])

        # Create discontinuity

        invalid_addition = torch.randint_like(pos, low=0, high=2)
        invalid_pos = pos + invalid_addition * mask.unsqueeze(2)

        pos_list.append(invalid_pos[:300])
        mask_list.append(mask[:300])
        validity_list.append(invalid_validity[:300])

    
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