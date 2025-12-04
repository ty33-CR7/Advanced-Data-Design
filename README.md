共有したいコードや既存コードに変更がある場合は、pushをお願いします。使い慣れていないと思うので、徐々にお互い慣れていきましょう。
## プロジェクトのディレクトリ構造
.
├── code
│   ├── preprocessing
│   └── src
├── data
│   ├── CIFAR10
│   │   ├── BF
│   │   ├── CWALDP
│   │   ├── Rappor
│   │   └── raw
│   │      
│   └── FashionMNIST
│       ├── BF
│       ├── CWALDP
│       ├── Rappor
│       └── raw
├── experiment
├── result
│   ├── CIFAR10
│   │   ├── CWALDP
│   │   │   ├── CNN
│   │   │   ├── RF
│   │   │   └── Resnet
│   │   ├── Rappor
│   │   │   ├── CNN
│   │   │   ├── RF
│   │   │   └── Resnet
│   │   └── ZW+24
│   │       ├── CNN
│   │       ├── RF
│   │       └── Resnet
│   └── FashionMNIST
│       ├── CWALDP
│       │   ├── CNN
│       │   ├── RF
│       │   └── Resnet
│       ├── Rappor
│       │   ├── CNN
│       │   ├── RF
│       │   └── Resnet
│       └── ZW+24
│           ├── CNN
│           ├── RF
│           └── Resnet
└── split_indices_full_gray
    ├── CIFAR10
    └── FashionMNIST