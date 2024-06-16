import  subprocess

output = subprocess.run(["bash", "knn_ner.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
# output = subprocess.run(["ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
print(type(output))
print(output)
print('code: ', output.returncode, 'stdout: ', output.stdout)
