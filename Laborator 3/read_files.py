def read_files(path):
    files = []
    with open(path, 'r') as file:
        for line in file:
            file_instructions = {}
            elems = line.split()
            file_instructions['name'] = elems[0].split('/')[-1]
            file_instructions['path'] = elems[0]
            file_instructions['noOfCommunities'] = int(elems[1])
            files.append(file_instructions)
    return files