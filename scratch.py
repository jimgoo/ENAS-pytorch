
      # if args.dataset == 'beijing':
        #     home = os.path.expanduser('~')
        #     f_train = os.path.join(home, 'amb-data/beijing_train.h5')
        #     f_test = os.path.join(home, 'amb-data/beijing_test.h5')
        #     target = 'pm2.5'
        #     df_train = pd.read_hdf(f_train)
        #     df_test = pd.read_hdf(f_test)
        

lass DeepNet(to.nn.Module):

    def __init__(self, genome, config, classification=True, loss_function='MSELoss'):
        super().__init__()

        # Sequence of layers in deep network.
        self.modules = []

        # Flag for classification: softmax on output.
        self.classification = classification

        # Flag for loss_function
        self.loss_function = loss_function

        # Recurrent networks are not supported currently.
        self.recurrent = False

        self.set_genome(genome, config)

    def forward(self, x, hidden):
        for module in self.modules:
            x, hidden = module.forward(x, hidden)

        # ToDo: Handle the case with mixed regression and classification
        if self.classification:
            if self.loss_function == 'NLLLoss':
                x = log_softmax(x, dim=1)
            elif self.loss_function == 'CrossEntropyLoss':
                pass
            else:
                pass
        else:
            if self.loss_function == 'BCELoss':
                x = sigmoid(x)
            elif self.loss_function == 'KLDivLoss':  # may not need logsoftmax
                x = log_softmax(x)
            else:
                pass

        return x, hidden

    def reset_hidden(self):
        for module in self.modules:
            module.reset_hidden()

    def detach(self):
        for module in self.modules:
            module.detach()

    def set_genome(self, genome, config: Config):
        self.genome = genome
        G = nx.DiGraph([connection for connection, gene in genome.connections.items() if gene.enabled])
        connections = genome.connections
        modules = genome.nodes

        assert len(config.genome_config.input_keys) == 1, "Deep networks only support all inputs as single layer"
        assert len(config.genome_config.output_keys) == 1, "Deep networks only support all outputs as single layer"
        assert nx.algorithms.dag.is_directed_acyclic_graph(G), "Deep networks only support DAGs"

        inputs = config.genome_config.input_keys[0]
        outputs = config.genome_config.output_keys[0]

        for input_key, output_key in window(nx.algorithms.traversal.dfs_preorder_nodes(G, inputs), n=2):
            assert (input_key, output_key) in connections, "Connection not found in genome"
            gene = connections[(input_key, output_key)]

            assert input_key in modules or input_key == inputs, "Input node not found in genome"
            input_module = modules[input_key]

            assert output_key in modules, "Output node not found in genome"
            output_module = modules[output_key]

            module = output_module.torch(input_module, output_key)

            self.modules += [module]

        # TODO: other activation functions on the output layer seem to cause severe performance degradation.
        self.modules[-1].activation = identity

        for index, module in enumerate(self.modules):
            setattr(self, 'l{}'.format(index), module.function)

    def get_genome(self):
        """
        Updates the genome with the learned parameters from PyTorch and returns it.
        """
        self.cpu()
        genome = deepcopy(self.genome)
        for module in self.modules:
            id_ = module.id_
            genome.nodes[id_].save_parameters(module.function.state_dict())
        genome.trained = True
        return genome











class DeepNet(to.nn.Module):

    def __init__(self, genome, config, classification=True, loss_function='MSELoss'):
        super().__init__()

        assert len(config.genome_config.input_keys) == 1, "Deep networks only support all inputs as single layer"
        assert len(config.genome_config.output_keys) == 1, "Deep networks only support all outputs as single layer"

        # Sequence of layers in deep network.
        self.modules = []

        # Flag for classification: softmax on output.
        self.classification = classification

        # Flag for loss_function
        self.loss_function = loss_function

        # Recurrent networks are not supported currently.
        self.recurrent = False

        # keys for inputs and outputs
        self.inputs = config.genome_config.input_keys[0]
        self.outputs = config.genome_config.output_keys[0]
        self.num_inputs = config.genome_config.num_inputs

        self.set_genome(genome)

    def forward(self, x, hidden):
        for module in self.modules:
            x, hidden = module.forward(x, hidden)

        # ToDo: Handle the case with mixed regression and classification
        if self.classification:
            if self.loss_function == 'NLLLoss':
                x = log_softmax(x, dim=1)
            elif self.loss_function == 'CrossEntropyLoss':
                pass
            else:
                pass
        else:
            if self.loss_function == 'BCELoss':
                x = sigmoid(x)
            elif self.loss_function == 'KLDivLoss':  # may not need logsoftmax
                x = log_softmax(x)
            else:
                pass

        return x, hidden

    def reset_hidden(self):
        for module in self.modules:
            module.reset_hidden()

    def detach(self):
        for module in self.modules:
            module.detach()

    def set_genome(self, genome):
        self.genome = genome
        G = nx.DiGraph([connection for connection, gene in genome.connections.items() if gene.enabled])
        connections = genome.connections
        modules = genome.nodes

        assert nx.algorithms.dag.is_directed_acyclic_graph(G), "Deep networks only support DAGs"

        for input_key, output_key in window(nx.algorithms.traversal.dfs_preorder_nodes(G, self.inputs), n=2):

            assert (input_key, output_key) in connections, "Connection not found in genome"
            gene = connections[(input_key, output_key)]

            assert input_key in modules or input_key == self.inputs, "Input node not found in genome"
            input_module = modules[input_key]

            assert output_key in modules, "Output node not found in genome"
            output_module = modules[output_key]

            module = output_module.torch(input_module, output_key)

            self.modules += [module]

        # TODO: other activation functions on the output layer seem to cause severe performance degradation.
        self.modules[-1].activation = identity

        for index, module in enumerate(self.modules):
            setattr(self, 'l{}'.format(index), module.function)

    def get_genome(self):
        """
        Updates the genome with the learned parameters from PyTorch and returns it.
        """
        self.cpu()
        genome = deepcopy(self.genome)
        for module in self.modules:
            id_ = module.id_
            genome.nodes[id_].save_parameters(module.function.state_dict())
        genome.trained = True
        return genome

