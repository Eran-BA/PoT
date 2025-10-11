flowchart TB
  %% Styles
  classDef head fill:#ffe0c2,stroke:#333,stroke-width:2px,color:#111
  classDef ctrl fill:#cfe8ff,stroke:#333,stroke-width:2px,color:#111
  classDef io fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#111
  classDef mix fill:#fff9c4,stroke:#333,stroke-width:2px,color:#111
  
  %% Nodes
  X[Input]:::io
  
  PC[Pointer Controller<br/>softmax over heads]:::ctrl
  
  subgraph SA[Self-Attention Heads ]
    direction LR
    H1[Head 1]:::head
    H2[Head 2]:::head
    H3[Head 3]:::head
  end
  
  MIX[Weighted/Selected<br/>Head Output]:::mix
  Y[Output]:::io
  
  %% Forward pass
  X --> PC
  PC -->|α₁| H1
  PC -->|α₂| H2
  PC -->|α₃| H3
  
  H1 --> MIX
  H2 --> MIX
  H3 --> MIX
  
  MIX --> Y
  
  %% Feedback cycles (recurrent control)
  H1 -.feedback.-> PC
  H2 -.feedback.-> PC
  H3 -.feedback.-> PC
  
  %% Optional recurrence over time
  Y -.next step context.-> PC

## Usage

Run the parser with the following command:

```bash
python ud_pointer_parser.py --epochs 2 --batch_size 8 --halting_mode entropy --max_inner_iters 3 --routing_topk 2
```