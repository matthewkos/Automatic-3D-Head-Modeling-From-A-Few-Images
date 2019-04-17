def read_hair_obj(head_obj,save_name,head_new):
    f = open(head_obj,'r')
    obj = f.read().split('\n')
    hair_end = obj.index('o Face_Face-mesh')

    with open(save_name,'w') as fout:
        for item in obj[:hair_end]:
            fout.write("%s\n" % item)
        for item in head_new: 
            fout.write("%s\n" % item)

read_hair_obj('test.obj','out.obj',['line 1','line 2'])
    
