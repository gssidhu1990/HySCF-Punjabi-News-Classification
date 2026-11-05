#!/usr/bin/env python3
"""Seed-42 reviewer model: find Kaggle checkpoint, export flat release, predict.

Examples:
  python code/06_seed42_model.py find
  python code/06_seed42_model.py export --source-root /kaggle/working/HySCF_SixFusion_5Seed_Components
  python code/06_seed42_model.py check
  python code/06_seed42_model.py predict --title "..." --content "..."

The original training notebook saves into three directories: experiments,
preprocessors, protocol.  This utility copies the authentic model/preprocessors
into the one flat GitHub experiments/ folder. Nothing is trained or refitted.
Use only trusted author-owned pickle/vectorizer artifacts.
"""
import argparse, csv, hashlib, json, os, pickle, re, shutil, sys, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO/'experiments'
MODEL_ID = 'F06_PARALLEL_FLAT_CNN'
SEED=42
CORE = {'model.keras','tokenizer.json','tfidf_vectorizer.pkl','label_mapping.json',
        'MODEL_CARD.json','model_architecture.json'}


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_files(root):
    root=Path(root)
    ex=root/'experiments'/f'{MODEL_ID}__seed{SEED}'
    pre=root/'preprocessors'
    return {
      'model.keras':ex/'model.keras',
      'MODEL_CARD.json':ex/'MODEL_CARD.json',
      'model_architecture.json':ex/'model_architecture.json',
      'fit_metrics.json':ex/'fit_metrics.json',
      'test_metrics.json':ex/'test_metrics.json',
      'history.csv':ex/'history.csv',
      'best.weights.h5':ex/'best.weights.h5',
      'test_probabilities.npz':ex/'test_probabilities.npz',
      'tokenizer.json':pre/'tokenizer.json',
      'tfidf_vectorizer.pkl':pre/'tfidf_5000'/'vectorizer.pkl',
      'label_mapping.json':pre/'label_mapping.json',
    }


def check():
    missing=[n for n in CORE if not (OUT/n).is_file()]
    print('Seed 42 model release folder:',OUT)
    if missing:
        print('MISSING: '+', '.join(sorted(missing)))
        return False
    card=json.loads((OUT/'MODEL_CARD.json').read_text(encoding='utf-8'))
    if int(card.get('seed',-1)) !=42 or int(card.get('tfidf_features',-1))!=5000 or int(card.get('parameters',-1))!=3417100:
        raise ValueError('Model card is not the original selected 5k seed-42 model.')
    if not zipfile.is_zipfile(OUT/'model.keras'):
        raise ValueError('model.keras is not a full Keras model ZIP.')
    with zipfile.ZipFile(OUT/'model.keras') as z:
        names=z.namelist()
        if 'config.json' not in names or not any(n.endswith(('weights.h5','weights.npz')) for n in names):
            raise ValueError('Model archive lacks architecture/weights.')
    vector=OUT/'tfidf_vectorizer.pkl'
    if not vector.stat().st_size:raise ValueError('Empty vectorizer.')
    classes=json.loads((OUT/'label_mapping.json').read_text(encoding='utf-8'))['classes']
    expected=json.loads((REPO/'dataset'/'label_mapping.json').read_text(encoding='utf-8'))['classes']
    if classes!=expected:raise ValueError('Class map mismatch.')
    print('CHECK PASSED: native files present; no actual forward inference has run yet.')
    return True


def export(root, force=False):
    files=source_files(root)
    missing=[str(p) for n,p in files.items() if n in CORE and not p.is_file()]
    if missing:raise FileNotFoundError('Export refused; native files missing:\n'+'\n'.join(missing))
    card=json.loads(files['MODEL_CARD.json'].read_text(encoding='utf-8'))
    if int(card.get('seed',-1))!=42 or int(card.get('tfidf_features',-1))!=5000 or int(card.get('parameters',-1))!=3417100:
        raise ValueError('Wrong seed, feature size or parameter count; refusing mixed release.')
    model=files['model.keras']
    if not zipfile.is_zipfile(model):raise ValueError('Not a full Keras model')
    with zipfile.ZipFile(model) as z:
        if 'config.json' not in z.namelist() or not any(n.endswith(('weights.h5','weights.npz')) for n in z.namelist()):
            raise ValueError('model.keras does not contain full architecture and weights')
    if not force:
        collisions=[n for n in files if (OUT/n).exists()]
        if collisions:raise FileExistsError('Target already has files; refusing overwrite: '+', '.join(collisions))
    OUT.mkdir(parents=True,exist_ok=True)
    audit={}
    for name,src in files.items():
        if not src.is_file():continue
        dest=OUT/name
        shutil.copy2(src,dest)
        if sha256(src)!=sha256(dest):raise IOError('Copy checksum mismatch for '+name)
        audit[name]={'sha256':sha256(dest),'bytes':dest.stat().st_size}
    (OUT/'seed42_release_manifest.json').write_text(json.dumps({'source':str(root),'seed':42,
        'architecture':'F06_PARALLEL_FLAT_CNN','files':audit,
        'publisher_full_text_included':False},indent=2),encoding='utf-8')
    check()
    print('COPIED',len(audit),'real model/preprocessor files into',OUT)
    print('If GitHub refuses a >100 MB file, use Git LFS or a release asset.')


def find(roots):
    count=0
    for base in roots:
        base=Path(base)
        if not base.exists():continue
        for path in base.rglob(f'{MODEL_ID}__seed42/MODEL_CARD.json'):
            root=path.parent.parent.parent
            files=source_files(root)
            missing=[k for k in CORE if not files[k].is_file()]
            print('NATIVE ROOT:',root,'READY' if not missing else 'MISSING: '+', '.join(missing))
            count+=1
    if not count:
        print('No native seed-42 checkpoint found at the given roots. RESULTS_ONLY ZIP does not contain model weights.')


def make_input(texts,tok):
    import numpy as np
    out=np.zeros((len(texts),15,30),dtype=np.int32)
    for i,text in enumerate(texts):
        chunks=re.split(r'[।!?]',text)[:15]
        for j,seq in enumerate(tok.texts_to_sequences(chunks)):
            seq=seq[:30]
            out[i,j,:len(seq)]=seq
    return out


def predict(title,content):
    if not check():raise FileNotFoundError('Export the native seed42 artifacts first.')
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    text=(str(title).strip()+' '+str(content).strip()).strip()
    tok=tokenizer_from_json((OUT/'tokenizer.json').read_text(encoding='utf-8'))
    with (OUT/'tfidf_vectorizer.pkl').open('rb') as f:vec=pickle.load(f)
    model=tf.keras.models.load_model(OUT/'model.keras',compile=False)
    names={x.name.split(':')[0] for x in model.inputs}
    h=make_input([text],tok)
    inputs={'flat_input':h.reshape(1,-1)} if 'flat_input' in names else {'han_input':h}
    if 'tfidf_input' in names:
        inputs['tfidf_input']=vec.transform([text]).toarray().astype(np.float32)
    elif 'tfidf_input' not in names:
        raise ValueError('Selected three-branch model must have TF-IDF input.')
    prob=np.asarray(model(inputs,training=False),dtype=np.float32)
    classes=json.loads((OUT/'label_mapping.json').read_text(encoding='utf-8'))['classes']
    if prob.shape!=(1,len(classes)) or not np.isfinite(prob).all():raise ValueError('Invalid output shape')
    if not np.isclose(prob.sum(),1,atol=1e-4):raise ValueError('Probabilities do not sum to one')
    pred=int(prob.argmax(1)[0])
    print(json.dumps({'label':classes[pred],'confidence':float(prob[0,pred]),
       'probabilities':{classes[j]:float(p) for j,p in enumerate(prob[0])}},ensure_ascii=False,indent=2))


def report():
    for fn in ('TEST_MEAN_SD.csv','ablation_FIVESEED_MEAN_SD.csv',
               'lexical_ALL_7_MODELS_MEAN_SD_5000_10000.csv'):
        p=OUT/fn
        if not p.exists():print('MISSING:',fn);continue
        with p.open(encoding='utf-8-sig',newline='') as f:
            rdr=csv.reader(f);rows=list(rdr)
        print('\n',fn,':',max(0,len(rows)-1),'rows; columns:',', '.join(rows[0][:8]))
    manifest=REPO/'dataset'/'fixed_primary_manifest.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    counts={}
    for r in records:counts[r['partition']]=counts.get(r['partition'],0)+1
    if len(records)!=25881 or counts!={'train':20703,'val':2589,'test':2589}:
        raise ValueError('Unexpected frozen dataset split manifest: '+str(counts))
    print('Split manifest:',len(records),'rows',counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='action',required=True)
    a=sp.add_parser('find');a.add_argument('--roots',nargs='+',default=['/kaggle/working','/kaggle/input'])
    a=sp.add_parser('export');a.add_argument('--source-root',required=True);a.add_argument('--force',action='store_true')
    sp.add_parser('check');sp.add_parser('report')
    a=sp.add_parser('predict');a.add_argument('--title',required=True);a.add_argument('--content',required=True)
    x=p.parse_args()
    if x.action=='find':find(x.roots)
    elif x.action=='export':export(x.source_root,x.force)
    elif x.action=='check':sys.exit(0 if check() else 1)
    elif x.action=='predict':predict(x.title,x.content)
    else:report()
if __name__=='__main__':main()
