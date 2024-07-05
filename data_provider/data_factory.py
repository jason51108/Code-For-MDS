from data_provider.data_loader import Binomial_data, Poisson_data, Normal_data, Custom_data

data_dict = {
    'Binomial': Binomial_data,
    'Poisson': Poisson_data,
    'Normal': Normal_data,
    'Custom': Custom_data
}


def data_provider(args):
    # choose data
    if args.data == 'Simulation':
        Data = data_dict[args.model]
    elif args.data == 'Custom':
        Data = data_dict['Custom']
    
    if args.task_name == 'parameter estimation':
        result = Data(args)

        
    elif args.task_name == 'classification':
        pass

    elif args.task_name == 'matrix completion':
        pass
    
    return result
