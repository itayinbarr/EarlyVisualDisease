# Primary Vision Pathways Disease Modeling

## Background and Motivation

The human visual system is a complex network of neural pathways that enable us to perceive and interpret the world around us. Understanding how visual information is processed in the early stages of the visual system is crucial for comprehending the underlying mechanisms of visual perception and the potential impact of various diseases on visual function.

Modeling early vision pathways and their associated diseases provides valuable insights into the workings of the visual system and aids in the development of diagnostic tools and treatment strategies. By simulating the behavior of the early visual system under different conditions, we can gain a deeper understanding of how diseases affect visual processing and identify potential interventions to mitigate their impact.

## Itti-Koch Saliency Model

The Itti-Koch saliency model, proposed by Laurent Itti and Christof Koch, is a prominent computational model that closely mimics the features of early primate vision. This model has been widely recognized for its ability to accurately predict human fixation patterns and capture the essential aspects of visual attention.

The Itti-Koch model is particularly well-suited for modeling early vision pathways due to its biologically plausible architecture and its incorporation of key features found in the early stages of visual processing. As stated in the original paper, "The model is based on the biologically plausible architecture proposed by Koch and Ullman (1985), in which early visual features are combined into a single topographical saliency map" (Itti, Koch, & Niebur, 1998).

The model consists of several stages that mimic the hierarchical processing of visual information in the primate visual system. It begins by extracting low-level features such as intensity, color, and orientation from the input image. These features are then processed through a series of center-surround operations, which highlight salient regions in the image. The resulting feature maps are normalized and combined into a single saliency map that represents the most visually prominent areas of the image.

## Base Model and Disease Model

In this project, we have implemented the Itti-Koch saliency model as a base model for modeling early vision pathways. The base model, represented by the `BaseVisualAttention` class, serves as the core functionality of the Itti-Koch model and provides a foundation for extending it to model specific diseases.

The `DiseaseModel` class is derived from the `BaseVisualAttention` class and serves as a template for modeling various diseases that affect early vision pathways. It inherits the basic functionality of the Itti-Koch model and allows for customization and parameterization to simulate the impact of different diseases on visual attention.

The relationship between the base model and the disease model is hierarchical, with the disease model building upon the functionality provided by the base model. This modular architecture enables researchers to easily extend the model to incorporate new diseases by creating subclasses of the `DiseaseModel` class and specifying the relevant parameters and modifications.

## Running Experiments

To run experiments of existing diseases using the early vision pathways disease modeling framework, follow these steps:

1. Create a new disease model by extending the `DiseaseModel` class in a separate file within the `experiments` directory. Specify the theoretical and empirical weights for the disease and implement any necessary modifications to the model.

2. Run the `ExperimentRunner.py` script, specifying the directory containing the input images and the desired output directory for the results.

The experiment runner will iterate over each image and each disease model, computing the saliency maps and saving the results in the specified output directory.

## Adding a New Disease

To add a new disease to the early vision pathways disease modeling framework, follow these steps:

1. Create a new file in the `experiments` directory for your disease model (e.g., `GlaucomaModel.py`).

2. Define the theoretical and empirical weights specific to your disease based on research findings and experimental data.

3. Create a new class that extends the `DiseaseModel` class and implement any necessary modifications or additions to the model to simulate the impact of the disease on visual attention.

4. Implement a function to run the experiment for your disease model, similar to the existing `run_macular_degeneration_experiment` and `run_control_experiment` functions.

5. Update the `ExperimentRunner.py` script to import your newly created disease model and add it to the list of experiments to be run.

By following these steps, you can easily integrate new diseases into the framework and conduct experiments to study their impact on early vision pathways.

## Conclusion

Modeling early vision pathways and their associated diseases provides valuable insights into the functioning of the visual system and the potential effects of various disorders on visual perception. The Itti-Koch saliency model serves as a biologically plausible foundation for simulating early visual processing, and the modular architecture of the base model and disease model allows for easy extension and customization to incorporate new diseases.

By utilizing this framework, researchers can conduct experiments, generate saliency maps, and analyze the impact of different diseases on visual attention. The results obtained from these experiments can contribute to a better understanding of the underlying mechanisms of visual perception and aid in the development of diagnostic tools and treatment strategies for visual disorders.

## References

Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254-1259.

Koch, C., & Ullman, S. (1985). Shifts in selective visual attention: towards the underlying neural circuitry. Human Neurobiology, 4(4), 219-227.
