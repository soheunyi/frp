import pathlib
import pickle

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.absolute()
PARAMS_PATH = CURRENT_FILE_PATH / "data/params/"
EXP_INFOS_PATH = CURRENT_FILE_PATH / "data/results/"


def get_filenames(query: dict = None):
    all_filenames = list([file.name for file in EXP_INFOS_PATH.glob("*.pkl")])
    if query is None:
        return all_filenames
    else:
        filenames = []
        for filename in all_filenames:
            with open(EXP_INFOS_PATH / filename, "rb") as f:
                exp_info = pickle.load(f)

            is_valid = True
            for key, value_list in query.items():
                if (
                    key not in exp_info["params"]
                    or exp_info["params"][key] not in value_list
                ):
                    is_valid = False
                    break

            if is_valid:
                filenames.append(filename)

        return filenames


def prune_data():
    exp_info_filenames = get_filenames()
    param_filenames = [file.name for file in PARAMS_PATH.glob("*.pkl")]

    keep_filenames = list(set(param_filenames) & set(exp_info_filenames))

    # # prune params if not in results
    for filename in list(set(param_filenames) - set(keep_filenames)):
        print("Deleting ", PARAMS_PATH / filename)
        (PARAMS_PATH / filename).unlink()


def load_file(filename):
    with open(EXP_INFOS_PATH / filename, "rb") as f:
        exp_info = pickle.load(f)

    return exp_info


def load_exp_info(query: dict):
    filenames = get_filenames(query)
    return [load_file(filename) for filename in filenames]


def delete_tag(tag):
    """
    WARNING: This function deletes all files with the given tag.
    """
    delete_targets = get_filenames({"tag": [tag]})

    for filename in delete_targets:
        if (PARAMS_PATH / filename).exists():
            print("Deleting ", PARAMS_PATH / filename)
            (PARAMS_PATH / filename).unlink()
        if (EXP_INFOS_PATH / filename).exists():
            print("Deleting ", EXP_INFOS_PATH / filename)
            (EXP_INFOS_PATH / filename).unlink()

    prune_data()


def get_tag_list():
    filenames = get_filenames()
    tags = []

    for filename in filenames:
        with open(EXP_INFOS_PATH / filename, "rb") as f:
            par = pickle.load(f)["params"]
            tag = par.get("tag")
            if tag not in tags:
                tags.append(tag)

    return tags


if __name__ == "__main__":
    prune_data()
    print(get_tag_list())
