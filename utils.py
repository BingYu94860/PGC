import numpy as np
import scipy.sparse as sp
import datetime

def get_nvidia_gpu_name(verbose=False):
    import subprocess
    sP = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sP.communicate()[0].decode()
    if verbose:
        print(out_str, end='')
    first_line = out_str.split('\n')[0]
    str_list = first_line.split()
    # ['Tesla K80' <= 'Tesla P4' < 'Tesla T4' <= 'Tesla P100-PCIE-16GB']
    if 'failed' not in str_list:
        return " ".join(str_list[2:4])
    else:
        return "No GPU"

#==============================================================================#
def get_now_time_str():
    return (datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime("%Y:%m:%d_%H:%M:%S")

#==============================================================================#
# 取出無向圖的邊
def get_edges(sparse_matrix, is_triu=True):
    coo = sp.coo_matrix(sparse_matrix)
    if is_triu:
        coo = sp.triu(coo, 1)
    return np.vstack((coo.row, coo.col)).transpose()#.tolist()

# 從邊組成無項圖的鄰接矩陣
def get_adj(edges, num_nodes):
    e_rows, e_cols = np.array(edges, dtype=np.int).transpose()
    values = np.ones(shape=[len(e_rows),], dtype=np.float32)
    adj = sp.coo_matrix((values, (e_rows, e_cols)), shape=[num_nodes,num_nodes])
    # triu adj --> adj
    adj.setdiag(0)
    bigger = adj.T > adj
    adj = adj - adj.multiply(bigger) + adj.T.multiply(bigger)
    return adj