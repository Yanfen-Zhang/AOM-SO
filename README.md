# AOM-SO
This is the source code of the paper: "Adaptive-Oriented Mutation Snake Optimizer for Scheduling Budget-Constrained Workflows in Heterogeneous Cloud Environments". A brief introduction of this work is as follows:

> Cloud computing, recognized as an advanced computing paradigm, facilitates flexible and efficient resource management and service delivery through virtualization and resource sharing. However, the computational capabilities of resources in heterogeneous cloud environments are often correlated with their costs; thus, budget constraints are imposed on users who require rapid response times. We introduce a novel metaheuristic optimization algorithm called the snake optimizer (SO), which is aimed at workflow scheduling in cloud environments, to tackle the challenge mentioned. We also integrate random mutation to enhance the algorithm’s global search capability to overcome the limitation of SO’s being prone to local optima. Additionally, we aim to increase the success rate of finding feasible solutions within budget constraints; thus, we implement a directional strategy to guide the evolutionary paths of the snake individuals. In this context, excessive randomness and overly rigid directionality can adversely affect the algorithm’s search performance. We propose an adaptive-oriented mutation (AOM) mechanism to balance the two aspects mentioned. This AOM mechanism is integrated with SO to create AOM-SO, which effectively addresses the makespan minimization problem for workflow scheduling under budget constraints in heterogeneous cloud environments. Comparative experiments using real-world scientific workflows show that AOM-SO achieves a 100% success rate in identifying feasible solutions. Moreover, compared with the state-of-the-art algorithms, it reduces makespan by an average of 43.58%.

This work was submitted to Future Generation Computer Systems. We note that a shorter version of this paper was accepted at the IEEE ISPA conference in 2024. Click [here](10.1109/ISPA63168.2024.00046) for our conference paper online.

# Usage
- `AOM_SO.py`  # The AOM-SO algorithm  
- `TestDAG.py`  # A typical workflow application

To start the project, run:  
```bash
python TestDAG.py
