import os


if __name__ == '__main__':
    non_compliant = []
    compliant = []
    with open("../LICENSE", "r") as licensefile:
        license = licensefile.readlines()
        for root, dirs, files in os.walk(".."):
            for file in files:
                if ".ci" in root or not file.endswith(".py"):
                    continue
                with open(os.path.join(root, file), "r") as f:
                    content = f.readlines()
                    if len(content) < len(license):
                        continue
                    is_compliant = True
                    for i in range(len(license)):
                        if license[i] not in content[i]:
                            is_compliant = False
                            break
                    if is_compliant:
                        compliant.append(os.path.join(root, file))
                    else:
                        non_compliant.append(os.path.join(root, file))

    assert len(non_compliant) == 0, "The following files do not contain the correct license: {}".format(non_compliant)
