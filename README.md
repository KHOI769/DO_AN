# Hướng dẫn sử dụng
*Đầu tiên bạn cần truy cập vào google colab và tạo một dự án mới và copy mã từ file demo_sigopt.py.

**B1: IMPORT THƯ VIỆN SIGOPT**

!pip install sigopt

**B2: LẤY API TOKEN BẰNG CÁCH BẠN CẦN ĐĂNG NHẬP VÀO SOGOPT VÀ COPY MÃ API TOKENS CỦA MÌNH CÓ MÃ NHƯ SAU: JIGCTDCWFICIUSWUHDJIGFUUKSTRLWMCKRJXBDIDQILQTVFW, VÀ SỬ DỤNG ĐOẠN MÃ SAU ĐỂ LIÊN KẾT VỚI SIGOPT.**

import sigopt

%load_ext sigopt

%sigopt config

**B3: IMPORT MỘT SỐ THƯ VIỆN CẦN THIẾT **

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn import datasets

import numpy

import sigopt

import time

**B4: TẢI TẬP DỮ LIỆU**

DATASET_NAME = "Sklearn Wine"

FEATURE_ENG_PIPELINE_NAME = "Sklearn Standard Scalar"

PREDICTION_TYPE = "Multiclass"

DATASET_SRC = "sklearn.datasets"


def get_data():

  """
  Load sklearn wine dataset, and scale features to be zero mean, unit variance.
  One hot encode labels (3 classes), to be used by sklearn OneVsRestClassifier.
  """

  data = datasets.load_wine()
  X = data["data"]
  y = data["target"]

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  enc = OneHotEncoder()
  Y = enc.fit_transform(y[:, numpy.newaxis]).toarray()

  return (X_scaled, Y)
  
**B5: TẠO HÀM CHỨC NĂNG CHO MÔ HÌNH**

MODEL_NAME = "OneVsRestClassifier(XGBoostClassifier)"

def evaluate_xgboost_model(X, y,

                           number_of_cross_val_folds=5,
                           max_depth=6,
                           learning_rate=0.3,
                           min_split_loss=0):
    t0 = time.time()
    classifier = OneVsRestClassifier(XGBClassifier(
        objective = "binary:logistic",
        max_depth =    max_depth,
        learning_rate = learning_rate,
        min_split_loss = min_split_loss,
        use_label_encoder=False,
        verbosity = 0
    ))
    cv_accuracies = cross_val_score(classifier, X, y, cv=number_of_cross_val_folds)
    tf = time.time()
    training_and_validation_time = (tf-t0)
    return numpy.mean(cv_accuracies), training_and_validation_time
    
**B6: TẠO HÀM THEO DÕI VÀ GHI LẠI THÔNG TIN MÔ HÌNH**

def run_and_track_in_sigopt():

    (features, labels) = get_data()

    sigopt.log_dataset(DATASET_NAME)
    sigopt.log_metadata(key="Dataset Source", value=DATASET_SRC)
    sigopt.log_metadata(key="Feature Eng Pipeline Name", value=FEATURE_ENG_PIPELINE_NAME)
    sigopt.log_metadata(key="Dataset Rows", value=features.shape[0]) 
    sigopt.log_metadata(key="Dataset Columns", value=features.shape[1])
    sigopt.log_metadata(key="Execution Environment", value="Colab Notebook")
    sigopt.log_model(MODEL_NAME)

    sigopt.params.setdefault("max_depth", numpy.random.randint(low=3, high=15, dtype=int))
    sigopt.params.setdefault("learning_rate", numpy.random.random(size=1)[0])
    sigopt.params.setdefault("min_split_loss", numpy.random.random(size=1)[0]*10)

    args = dict(X=features,
                y=labels,
                max_depth=sigopt.params.max_depth,
                learning_rate=sigopt.params.learning_rate,
                min_split_loss=sigopt.params.min_split_loss)

    mean_accuracy, training_and_validation_time = evaluate_xgboost_model(**args)

    sigopt.log_metric(name='accuracy', value=mean_accuracy)
    sigopt.log_metric(name='training and validation time (s)', value=training_and_validation_time)
    
**B7: TỐI ƯU HOÁ MÔ HÌNH**

%%experiment

{

    'name': 'XGBoost Optimization',
    'metrics': [
        {
            'name': 'accuracy',
            'strategy': 'optimize',
            'objective': 'maximize',
        }
    ],
    'parameters': [
        {
            'name': 'max_depth',
            'type': 'int',
            'bounds': {'min': 3, 'max': 12}
        },
        {
            'name': 'learning_rate',
            'type': 'double',
            'bounds': {'min': 0.0, 'max': 1.0}
        }
    ],
    'budget': 4
}

**B8: CHẠY CHƯƠNG TRÌNH**

%%optimize teamdemo

run_and_track_in_sigopt()


  




