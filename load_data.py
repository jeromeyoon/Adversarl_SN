import glob,pickle,os.path,pdb
from sorting import natsorted
def load_train_val_files():
    datapath ='/research2/ECCV_dataset_resized'
    val_data = [11,16,21,22,33,36,38,53,59,92]
    dataset={}
    train_input=[]
    train_gt=[]
    val_input=[]
    val_gt=[]
    folders = glob.glob(os.path.join(datapath,'*'))
    folders = natsorted(folders)
    count =1
    for ff in range(len(folders)):
        subfolders  = glob.glob(os.path.join(datapath,folders[ff],'*'))
        subfolders = natsorted(subfolders)
        for ss in range(len(subfolders)):
	    print('folder:%d subfolder %d \n' %(ff,ss))
            files = glob.glob(os.path.join(datapath,folders[ff],subfolders[ss],'*.bmp'))
    	    files = natsorted(files)
	    if count not in val_data:
	        train_input.append(files[:-1])
		train_gt.append(files[-1])
            else:
	        val_input.append(files[:-1])
		val_gt.append(files[-1])
	count += 1		 
    dataset['train_input'] = train_input
    dataset['train_gt'] = train_gt
    dataset['val_input'] = val_input
    dataset['val_gt']= val_gt
    with open('ECCV_256size.pickle','wb') as f:
        pickle.dump(dataset,f)

def load_pickle():
	
    with open('ECCV_256size.pickle','rb') as f:
        dataset = pickle.load(f)
	return dataset
"""
if __name__=='__main__':
    load_pickle()
"""
