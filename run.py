from main import run_ReguSync

run_ReguSync(
    n_epochs=200,
    train_batch_size=128,
    test_batch_size=256,

    lr=0.001,
    folds=5,

    modal_a_train="./Dataset/Paired_RNA_train.h5ad",
    modal_b_train="./Dataset/Paired_ATAC_train.h5ad",
    modal_a_test="./Dataset/Paired_RNA_test.h5ad",
    modal_b_test="./Dataset/Paired_ATAC_test.h5ad",

    modal_a="RNA",
    modal_b="ATAC",

    modal_a_loss="nb",
    modal_b_loss="nb",

    species="human",
    run=True,
    seed=2026,
    save_embds=True,

    max_seq_len=1600,
    n_layers=4,
    n_bins=50,
    d_model=128,
    n_hvg=1000,

    hvg_flavor="seurat_v3",
    hvg_flavor_2="cell_ranger",

    ram_usage_optimization=False,
    patience=100,

    evaluation_a="NMI",
    evaluation_b="NMI",

    spatial=False,
    have_labels=True,
    first_run=False,
)