import os

def _rename(path):
    file_list = os.listdir(path)
    total_num = len(file_list)
    i = 1
    for item in file_list:
        if item.endswith(".png"):
            src = os.path.join(os.path.abspath(path),item)
            dst = os.path.join(os.path.abspath(path),str(i)+'.png')
            try:
                os.rename(src=src,dst=dst)
                print('>>>>')
                i = i +1
            except:
                continue

    print("convert {} photos in total".format(total_num))

path = './negative_ROI/'
_rename(path)