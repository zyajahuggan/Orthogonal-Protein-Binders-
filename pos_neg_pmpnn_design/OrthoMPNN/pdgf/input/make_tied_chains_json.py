import argparse
import json

# written for vanilla PMPNN
def main(args):
    with open(args.input_path) as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)

    my_dict = {}
    tied_positions_list = []

    """
    for resi in intf_resi['A']:
        temp_dict = {}
        temp_dict['A'] = [[resi], [1.0]]
        temp_dict['D'] = [[resi], [-0.5]]
        tied_positions_list.append(temp_dict)

    for resi in intf_resi['B']:
        temp_dict = {}
        temp_dict['B'] = [[resi], [1.0]]
        temp_dict['E'] = [[resi], [-0.5]]
        tied_positions_list.append(temp_dict)

    for resi in intf_resi['C']:
        temp_dict = {}
        temp_dict['C'] = [[resi], [1.0]]
        temp_dict['I'] = [[resi], [-0.5]]
        tied_positions_list.append(temp_dict)
    """

    for i in range(1,len(result[f"seq_chain_{'A'}"])+1):
        temp_dict = {}
        temp_dict['A'] = [i]
        temp_dict['B'] = [i]
        tied_positions_list.append(temp_dict)

    for i in range(1,len(result[f"seq_chain_{'C'}"])+1):
        temp_dict = {}
        temp_dict['C'] = [i]
        temp_dict['D'] = [i]
        tied_positions_list.append(temp_dict)
    
    # print(tied_positions_list)

    my_dict[result['name']] = tied_positions_list
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    args = argparser.parse_args()
    main(args)
