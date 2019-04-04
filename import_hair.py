def read_hair_obj(hair_path,save_name):
    f = open(hair_path,'r')
    hair = f.read().split('\n')
    hair_end = hair.index('o Face_Face-mesh')

    with open(save_name,'w') as fout:
        for item in hair[:hair_end]:
            fout.write("%s\n" % item)

read_hair_obj('./test.obj','out.obj')
    
