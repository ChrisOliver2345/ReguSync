from regusync_main import run_ReguSync

run_ReguSync(
    n_epochs=200,
    train_batch_size=128,
    test_batch_size=256,
    dataset="RNA_ATAC_translation",
    modal_a_train="./Dataset/Paired_RNA_train.h5ad",
    modal_b_train="./Dataset/Paired_ATAC_train.h5ad",
    modal_a_test="./Dataset/Paired_RNA_test.h5ad",
    modal_b_test="./Dataset/Paired_ATAC_test.h5ad",
    species="human",
    d_model=128,
    n_hvg=1000,
    ram_usage_optimization=False,
    spatial=False,
)