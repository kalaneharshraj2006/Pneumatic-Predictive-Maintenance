
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; MODELS=ROOT/"models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
FEATURES=["supply_pressure_bar","chamber_pressure_bar","pressure_drop_bar","air_flow_l_min","actuation_time_s","temperature_c","vibration_rms","load_factor"]
CLASSES=["normal","air_leakage","valve_stiction","pressure_regulator_fault"]

def generate_fault_run(label, run_id, n=100, rng=None):
    """Equation-based illustrative sensor trace; not a physical Simscape solver."""
    rng=rng or np.random.default_rng()
    t=np.linspace(0,1,n); cycle=np.arange(n); env_temp=rng.normal(25,2); load=rng.uniform(.75,1.25)
    noise=lambda sd:rng.normal(0,sd,n)
    supply=6.0+noise(.08); chamber=5.25+.18*np.sin(2*np.pi*t)+noise(.10)
    response=.42+.025*np.sin(2*np.pi*t)+noise(.025); flow=12+noise(.55)
    temp=env_temp+.35*t+noise(.25); vibration=.18+noise(.025)
    if label=="air_leakage":
        severity=.25+.85*t; chamber-=.65*severity; flow+=1.8*severity
        temp+=.8*severity; vibration+=.04*severity; response+=.09*severity
    elif label=="valve_stiction":
        severity=.2+.8*t; response+=.20*severity+.035*np.sin(8*np.pi*t)
        vibration+=.13*severity+.035*np.abs(np.sin(8*np.pi*t))
        flow-=.9*severity; chamber-=.22*severity; temp+=.35*severity
    elif label=="pressure_regulator_fault":
        severity=.2+.8*t; supply-=.85*severity; chamber-=.58*severity
        flow-=1.4*severity; response+=.10*severity; temp+=.2*severity
    return pd.DataFrame({"run_id":run_id,"cycle":cycle,"time_norm":t,
        "supply_pressure_bar":supply,"chamber_pressure_bar":chamber,
        "pressure_drop_bar":supply-chamber,"air_flow_l_min":flow,
        "actuation_time_s":response,"temperature_c":temp,"vibration_rms":vibration,
        "load_factor":load,"fault_type":label})

def generate_rul_data(seed=8675309,n_runs=180):
    rng=np.random.default_rng(seed); rows=[]
    for run in range(n_runs):
        total=int(rng.integers(180,321)); bias=rng.normal(0,.05)
        tb=rng.normal(0,.8); vb=rng.normal(0,.025)
        for cyc in range(0,total,5):
            wear=cyc/total; supply=6+rng.normal(0,.07)
            chamber=5.25-.95*wear+rng.normal(0,.10)+bias
            flow=12+1.2*wear+rng.normal(0,.45); act=.42+.22*wear+rng.normal(0,.025)
            temp=25+tb+1.5*wear+rng.normal(0,.35); vib=.18+vb+.14*wear+rng.normal(0,.025)
            rows.append({"run_id":f"wear_{run:03d}","cycle":cyc,
                "supply_pressure_bar":supply,"chamber_pressure_bar":chamber,
                "pressure_drop_bar":supply-chamber,"air_flow_l_min":flow,
                "actuation_time_s":act,"temperature_c":temp,"vibration_rms":vib,
                "load_factor":1+rng.normal(0,.04),"wear_fraction":wear,"rul_cycles":max(total-cyc,0)})
    return pd.DataFrame(rows)

def main():
    rng=np.random.default_rng(20261028)
    sensor=pd.concat([generate_fault_run(c,f"{c}_{j:03d}",rng=rng) for c in CLASSES for j in range(60)],ignore_index=True)
    sensor.to_csv(OUT/"synthetic_sensor_data.csv",index=False)
    run_num=sensor.run_id.str.rsplit("_",n=1).str[-1].astype(int)
    train,test=sensor[run_num<45].copy(),sensor[run_num>=45].copy()
    clf=RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight="balanced",random_state=42)
    clf.fit(train[FEATURES],train.fault_type); pred=clf.predict(test[FEATURES])
    acc=accuracy_score(test.fault_type,pred); cm=confusion_matrix(test.fault_type,pred,labels=CLASSES)
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/"fault_confusion_matrix.csv")
    pd.DataFrame(classification_report(test.fault_type,pred,output_dict=True,zero_division=0)).T.to_csv(OUT/"fault_classification_report.csv")
    normal=train[train.fault_type=="normal"]
    iso=make_pipeline(StandardScaler(),IsolationForest(n_estimators=250,contamination=.08,random_state=42))
    iso.fit(normal[FEATURES]); scores=-iso.decision_function(test[FEATURES]); flags=iso.predict(test[FEATURES])==-1
    actual=(test.fault_type!="normal").astype(int).to_numpy()
    precision,recall,f1,_=precision_recall_fscore_support(actual,flags.astype(int),average="binary",zero_division=0)
    auc=roc_auc_score(actual,scores)
    an=test[["run_id","cycle","fault_type"]].copy()
    an["anomaly_score"]=scores; an["anomaly_predicted"]=flags; an["anomaly_actual"]=actual
    an.to_csv(OUT/"anomaly_test_predictions.csv",index=False)
    rul=generate_rul_data(); rul.to_csv(OUT/"synthetic_rul_data.csv",index=False)
    ids={f"wear_{i:03d}" for i in range(135)}
    rt,rv=rul[rul.run_id.isin(ids)],rul[~rul.run_id.isin(ids)]
    reg=RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=42,n_jobs=-1)
    reg.fit(rt[FEATURES],rt.rul_cycles); rp=np.maximum(0,reg.predict(rv[FEATURES]))
    mae=mean_absolute_error(rv.rul_cycles,rp); rmse=mean_squared_error(rv.rul_cycles,rp)**.5
    ro=rv[["run_id","cycle","rul_cycles"]].copy(); ro["rul_predicted_cycles"]=rp
    ro.to_csv(OUT/"rul_test_predictions.csv",index=False)
    joblib.dump(clf,MODELS/"fault_classifier.joblib"); joblib.dump(iso,MODELS/"anomaly_detector.joblib"); joblib.dump(reg,MODELS/"rul_regressor.joblib")
    # Plots
    plt.figure(figsize=(9,4.8))
    for label in CLASSES:
        ex=sensor[sensor.run_id==f"{label}_000"]
        plt.plot(ex.cycle,ex.chamber_pressure_bar,label=label.replace("_"," "))
    plt.xlabel("Simulated cycle"); plt.ylabel("Chamber pressure (bar)")
    plt.title("Synthetic pneumatic chamber pressure by condition"); plt.legend(ncol=2,fontsize=8); plt.tight_layout()
    plt.savefig(OUT/"pressure_trends.png",dpi=160); plt.close()
    plt.figure(figsize=(6.8,5.5)); plt.imshow(cm,interpolation="nearest"); plt.title("Fault classification confusion matrix")
    plt.colorbar(); ticks=np.arange(len(CLASSES)); plt.xticks(ticks,[x.replace("_","\\n") for x in CLASSES],fontsize=8)
    plt.yticks(ticks,[x.replace("_"," ") for x in CLASSES],fontsize=8); plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(OUT/"fault_confusion_matrix.png",dpi=160); plt.close()
    plt.figure(figsize=(7,4.5))
    for label in CLASSES: plt.hist(an.loc[an.fault_type==label,"anomaly_score"],bins=30,alpha=.45,label=label.replace("_"," "))
    plt.xlabel("Anomaly score (higher = more unusual)"); plt.ylabel("Test samples")
    plt.title("Anomaly scores on unseen synthetic runs"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT/"anomaly_scores.png",dpi=160); plt.close()
    plt.figure(figsize=(7,4.8)); plt.scatter(rv.rul_cycles,rp,alpha=.18,s=10)
    lim=max(float(rv.rul_cycles.max()),float(rp.max())); plt.plot([0,lim],[0,lim],linestyle="--")
    plt.xlabel("Actual simulated RUL (cycles)"); plt.ylabel("Predicted RUL (cycles)")
    plt.title("RUL estimation on held-out wear trajectories"); plt.tight_layout()
    plt.savefig(OUT/"rul_actual_vs_predicted.png",dpi=160); plt.close()
    metrics={"dataset":{"fault_sensor_rows":int(len(sensor)),"fault_runs":int(sensor.run_id.nunique()),
      "train_rows":int(len(train)),"test_rows":int(len(test)),"rul_rows":int(len(rul)),
      "rul_train_runs":135,"rul_test_runs":45,"features":FEATURES,"seed":20261028},
      "fault_classification":{"accuracy":float(acc),"test_runs_per_class":15,
      "classification_report":classification_report(test.fault_type,pred,output_dict=True,zero_division=0),
      "confusion_matrix_labels":CLASSES,"confusion_matrix":cm.tolist()},
      "anomaly_detection":{"precision":float(precision),"recall":float(recall),"f1":float(f1),"roc_auc":float(auc)},
      "rul_regression":{"mae_cycles":float(mae),"rmse_cycles":float(rmse)},
      "scope_note":"Synthetic-data proof of concept only; not a Simscape model or real-machine validation."}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
