"""Weighted reanalysis of the original three fixed neural architectures.

Five seeds; validation-weighted AUC stopping; no hyperparameter search.
No test scores are used in architecture, epoch, seed, or threshold selection.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer":True})
from sklearn.metrics import roc_auc_score
import joblib
import revision_analysis as ra


def build(name, n):
    L=keras.layers
    inp=keras.Input(shape=(n,))
    if name=='MLP':
        x=L.Dense(128,activation='relu')(inp);x=L.Dropout(.30)(x)
        x=L.Dense(64,activation='relu')(x);x=L.Dropout(.20)(x)
        out=L.Dense(1,activation='sigmoid')(x)
    elif name=='WideDeep':
        wide=L.Dense(1)(inp);x=L.Dense(128,activation='relu')(inp);x=L.Dropout(.30)(x)
        x=L.Dense(64,activation='relu')(x);deep=L.Dense(1)(x)
        out=L.Activation('sigmoid')(L.Add()([wide,deep]))
    else:
        x=L.Dense(128,activation='relu')(inp)
        for _ in range(3):
            skip=x;z=L.Dense(128,activation='relu')(x);z=L.Dropout(.30)(z);z=L.Dense(128)(z)
            x=L.Activation('relu')(L.Add()([skip,z]))
        out=L.Dense(1,activation='sigmoid')(x)
    model=keras.Model(inp,out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),loss='binary_crossentropy')
    return model


def train(name,X,y,w,seed,epochs=50,validation=None):
    keras.backend.clear_session();keras.utils.set_random_seed(seed)
    model=build(name,X.shape[1]);rng=np.random.default_rng(seed)
    best=-np.inf;best_epoch=epochs;best_weights=None;stale=0
    @tf.function(input_signature=[tf.TensorSpec([None,X.shape[1]],tf.float32),tf.TensorSpec([None],tf.float32),tf.TensorSpec([None],tf.float32)])
    def step(xb,yb,wb):
        with tf.GradientTape() as tape:
            probability=model(xb,training=True)
            loss=tf.reduce_mean(keras.losses.binary_crossentropy(yb[:,None],probability)*wb)
        gradients=tape.gradient(loss,model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss
    for epoch in range(epochs):
        ids=rng.permutation(len(y))
        for offset in range(0,len(y),64):
            ix=ids[offset:offset+64]
            step(X[ix],y[ix],w[ix])
        if validation is not None:
            xv,yv,wv=validation;p=model(xv,training=False).numpy().ravel()
            auc=roc_auc_score(yv,p,sample_weight=wv)
            if auc>best:
                best=auc;best_epoch=epoch+1;best_weights=model.get_weights();stale=0
            else: stale+=1
            if stale>=6: break
    if best_weights is not None: model.set_weights(best_weights)
    return model,best_epoch,float(best) if validation is not None else None


def run(panel):
    _,data=ra.load_data();d,nums=ra.get_panel(data,panel)
    dest=ra.OUT/panel/'neural';dest.mkdir(exist_ok=True,parents=True)
    manifest=pd.read_csv(ra.OUT/panel/'split_manifest.csv')
    d=d.merge(manifest[['SEQN','split']],on='SEQN',validate='one_to_one')
    tr=d[d.split=='train'];va=d[d.split=='validation'];te=d[d.split=='test']
    rows=[];results=[]
    for config,remove in ra.CONFIGS.items():
        cn=[c for c in nums if c not in remove];feats=cn+ra.CATEGORICAL
        pre=ra.make_preprocessor(cn)
        xt=pre.fit_transform(tr[feats]).astype('float32');xv=pre.transform(va[feats]).astype('float32');xe=pre.transform(te[feats]).astype('float32')
        yt=tr.diabetes.to_numpy(dtype='float32');yv=va.diabetes.to_numpy()
        wt=(tr.analysis_weight/tr.analysis_weight.mean()).to_numpy(dtype='float32');wv=va.analysis_weight.to_numpy()
        summaries=[]
        for name in ['MLP','WideDeep','ResNet']:
            aucs=[];epochs=[]
            for seed in range(5):
                model,best_epoch,auc=train(name,xt,yt,wt,seed,validation=(xv,yv,wv))
                aucs.append(auc);epochs.append(best_epoch)
                rows.append(dict(config=config,model=name,seed=seed,best_epoch=best_epoch,validation_auc=auc))
                pd.DataFrame(rows).to_csv(dest/'development_by_seed.csv',index=False)
            summaries.append(dict(config=config,model=name,validation_auc_mean=float(np.mean(aucs)),validation_auc_sd=float(np.std(aucs,ddof=1)),median_epoch=int(np.median(epochs))))
            print(panel,config,name,'validation AUC',np.mean(aucs),flush=True)
        best=sorted(summaries,key=lambda r:(-r['validation_auc_mean'],r['model']))[0]
        # Fixed final seed, not the highest-performing development seed.
        model,_,_=train(best['model'],xt,yt,wt,ra.SEED,epochs=best['median_epoch'])
        valp=model(xv,training=False).numpy().ravel();cal=ra.calibrate(valp,yv,wv,'probability')
        raw=model(xe,training=False).numpy().ravel().astype('float64');p=ra.apply_calibrator(cal,raw,'probability')
        for variant,prob in [('raw',raw),('calibrated',p)]:
            metrics=ra.metrics(te.diabetes,prob,te.analysis_weight)
            row=dict(config=config,model=best['model'],variant=variant,test_n=len(te),test_cases=int(te.diabetes.sum()),**metrics)
            ci,note=ra.bootstrap_metrics(te,prob,500);row['uncertainty_note']=note
            ci.to_csv(dest/f'{config}_{variant}_psu_resampling_intervals.csv',index_label='metric')
            results.append(row)
        model.save(dest/f'{config}_{best["model"]}.keras')
        joblib.dump(dict(preprocessor=pre,calibrator=cal,features=feats,threshold=.20),dest/f'{config}_preprocessing.joblib')
        pd.DataFrame(dict(SEQN=te.SEQN,outcome=te.diabetes,weight=te.analysis_weight,raw=raw,calibrated=p)).to_csv(dest/f'{config}_test_predictions.csv',index=False,float_format='%.17g')
        ra.save_plot(te,{'Uncalibrated':raw,'Recalibrated':p},dest/f'{config}_calibration.png')
        pd.DataFrame(results).to_csv(dest/'test_results.csv',index=False)
    ra.dump(dest/'status.json',dict(completed=True,seeds=[0,1,2,3,4],final_seed=ra.SEED,
                                   maximum_epochs=50,patience=6,batch_size=64,optimizer='Adam',learning_rate=.001,
                                   loss='survey-weighted binary crossentropy',hyperparameter_search=False,
                                   training_implementation='compiled GradientTape; weighted BCE mean over each batch; checked against Keras reference update',selection='mean weighted validation AUC; final epochs median of selected architecture',psu_disjoint=True,synthetic_data_used=False,panel=panel))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=ra.PANELS+['all'],default='all');a=p.parse_args()
    for panel in ra.PANELS if a.panel=='all' else [a.panel]: run(panel)
