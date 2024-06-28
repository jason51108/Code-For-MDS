from data_provider.data_loader import Binomial_data

data_dict = {
    'Binomial': Binomial_data
}


def data_provider(args):
    Data = data_dict[args.model]
    
    if args.task_name == 'parameter estimation':
        true_alpha, true_theta, adjacency_matrix = Data(args)
        
    elif args.task_name == 'classification':
        pass

    elif args.task_name == 'matrix completion':
        pass
    
    return true_alpha, true_theta, adjacency_matrix
