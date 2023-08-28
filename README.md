# Custom Self-Attention Layer with 1x1 2D Convolutions

Welcome to the documentation for our custom self-attention layer, meticulously designed to enhance your models with self-attention capabilities through the utilization of 1x1 convolutions. This custom layer aims to enrich your models with advanced self-attention capabilities while maintaining the ease of integration that Keras offers.

**Overview:**
This custom layer implementation draws its design principles from the influential SAGAN paper ([Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318v2on), which introduced pioneering concepts in self-attention. While originally implemented in PyTorch, we've taken the initiative to translate and adapt this mechanism for Keras (and TensorFlow), enriching it with unique modifications that further elevate its functionality.

**Key Features:**

1. **Versatile Usage:** This custom layer can seamlessly serve as the firts layer within a convolutional model or as an intermediary layer.
2. **Input Flexibility:** When `filters` specifications are omitted, the layer automatically mirrors the input dimensions. For example, if the input is `(batch_size, width, height, channels)` and no `filters` are specified, the output retains the same shape.
3. **Strategic Filtering:** Designed for optimal adaptability, when used as the model's first layer, the number of filters can be customized. This empowers the network to derive diverse insights from the input data before passing this to subsequent convolutional layers. For instance, specifying `filters=8` with an input shape of `(batch_size, 28, 28, 1)` results in an output shape of `(batch_size, 28, 28, 8)`.
4. **Activation Flexibility:** Our custom self-attention layer supports a range of activation functions, allowing you to tailor the layer's behavior to your needs. Choose from activation functions like `relu`, `tanh`, and more. The default activation is set to `linear`.
5. **Attention Control with Gamma:** The custom self-attention layer introduces the `gamma` parameter, which governs the influence of the attention mask on the outputs. While you can set an initial value for Gamma, its final value is learned during the model training process. This parameter provides a dynamic means of adjusting the attention mechanism's impact based on the specific needs of your model.
6. **Attention Score Retrieval:** By setting the `return_attention_scores` parameter to `True`, the custom layer returns the inputs after applying the attention mask and scaling by Gamma. This allows you to access and analyze the attention scores for deeper insights into the model's decision-making process.
7. **Multi-Head Attention Support:** The custom self-attention layer includes the `num_head` parameter, which mirrors the multihead-attention mechanism. Unlike the `filters` parameter, `num_head` is an internal representation and doesn't affect the output shape. 

**Acknowledgments:**
This implementation has drawn inspiration and learning from the community's contributions. Special thanks to the authors of:

https://github.com/grohith327/simplegan/blob/master/simplegan/layers/selfattention.py
https://github.com/MorvanZhou/mnistGANs/blob/main/sagan.py

These resources have greatly contributed to shaping our implementation.

**Usage:**
To integrate this custom self-attention layer into your model, follow these steps:

1. **Download Custom Layer:** Download the `custom_layer.py` file, containing the implementation of the custom self-attention layer.
2. **Import the Layer:** Import the custom layer module into your project using `from custom_layer import SelfAttention_Conv2D`.
3. **Instantiate the Layer:** Create an instance of the custom self-attention layer, specifying the desired number of filters, activation function, initial Gamma value, and whether to return attention scores.
4. **Incorporate into Model:** Use the instantiated layer as you would with any standard Keras layer.

**Example:**
For a comprehensive usage example, refer to the companion Jupyter notebook `SelfAttention_1x1_Conv2D.ipynb`, where we walk through step-by-step implementation and visualization of the custom self-attention layer.
