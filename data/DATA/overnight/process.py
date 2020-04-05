import re

'''
Dataset: OVERNIGHT
(Git Repo)
'''

def _stop_words(s):
    slist = ['in','the','of','for']
    return ' '.join([x for x in s.split() if x not in slist])

def get_fields():
    all_dicts = {}
    all_dicts['all']= {}

    subset = 'basketball'
    subset_dict = {}
    subset_dict['assists']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['blocks']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['turnovers']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['points']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['fouls']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['steals']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['rebounds']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['games']=['2', '3', '4', 'one', 'two', 'three']
    subset_dict['player']=['lebron james', 'kobe bryant']
    subset_dict['team']=['los angeles lakers','cleveland cavaliers']
    subset_dict['position']=['point guard','forward']
    subset_dict['season']=['2004', '2010']
    all_dicts[subset] = subset_dict

    fields = {}
    for subset in all_dicts.keys():
        dictionary = all_dicts[subset]
        fields[subset] = dictionary.keys()

    return fields, all_dicts

def _match_str(phrase, qu):
    return phrase in qu.split() or (' ' in phrase and phrase in qu) 

def anno_qu(qu, all_fields, dictionary):
    # clean unnecessary words
    ori = qu
    cleaned_list = [ t for t in qu.split() if t not in ['total','during','those','find','got'] ]
    qu = ' '.join(cleaned_list)

    # human knowledge [basketball]
    if qu.startswith('who '):
        qu = qu.replace('who ','player ')
    if qu.startswith('whos '):
        qu = qu.replace('whos ','player ')
    if qu.startswith('when '):
        qu = qu.replace('when ','season ')
    if qu.startswith('number of '):
        qu = qu.replace('number of ',' ', 1)
    if ' year ' in qu:
        qu = qu.replace(' year ',' season ')


    # delete question words
    if qu.startswith('how many') or qu.startswith('which') or qu.startswith('what'):
        qu = qu.replace('how many ','',1).replace('which ','',1).replace('what ','',1)
   

    # Match Head
    for head in all_fields:
        if head in qu:
            if qu.index(head)<15:
                qu = qu.replace(head,'<f0>',1).replace('<f0>s','<f0>')
               
                
    if '<f0>' not in qu:
        # delete due to non-compatible with SQL convertion
        pass
    else:
        # annotate conditions after head pinpointed
        idx = 1
        for cond_c in all_fields:
            # (annotate c appears)
            if _match_str(cond_c, qu):
                for v in dictionary[cond_c]:
                    if _match_str(v, qu):
                        qu = qu.replace(cond_c, '<f'+str(idx)+'> '+cond_c+' <eof>').replace(v, '<v'+str(idx)+'>')
                        idx += 1
         
        for cond_c in all_fields:   
            # (annotate v appears but c does not appear)
            for _v in dictionary[cond_c]:
                if _match_str(_v, qu):
                    qu = qu.replace(_v, '<f'+str(idx)+'> ' + cond_c + ' <eof> <v'+str(idx)+'>')
                    idx += 1
    sql = None
    return ori, qu, sql



if __name__ == '__main__':

    symbols = ['where','count','(',')','equal','and','less','greater','<f0>','<f1>','<v1>','<f2>','<v2>']
    subdomain = 'basketball'

    fields, all_dicts  = get_fields()
    all_fields = fields[subdomain]
    dictionary = all_dicts[subdomain]

    for dataset in ['train','test']:

        fully_anno = 0 
        cnt = 0
        new_qus = []
 
        filename = subdomain+'.paraphrases.'+dataset+'.examples'
        with open(filename, 'r') as f:
            ls = f.readlines()
            for l in ls:
                l = l.strip().strip('(').strip(')')
                if l.startswith('utterance'):
                    l = l.strip('utterance').strip().strip('\"').strip()
                    l = ' '.join(l.split())

                    # [basketball] typo correction 
                    l = l.replace('kob ','kobe ').replace(' la ',' los angeles ').replace('three','3')
                    if 'kobe' in l and 'bryant' not in l:
                        l = l.replace('kobe', 'kobe bryant')
                    
                    ori, qu, sql = anno_qu(l, all_fields, dictionary)
                    new_qus.append(qu)
                    cnt += 1

                else:
                    continue

        print('Total count: {0}'.format(cnt))

        with open(subdomain+'/new_'+dataset+'.qu', 'w') as qu_w:
            for qu in new_qus:
                qu_w.write(qu+'\n')

      

