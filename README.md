# Hướng dẫn sử dụng
B1: Bạn cần mở file demo_sigopt.py sao chép code và bạn cần đăng nhập vào google colab tạo dự án và dán đoạn mã vừa copy vào.

B2: Đầu cần chạy đoạn mã cài đặt thư viện sigopt cho dự án.

!pip install sigopt

B3:Lấy API_TOKEN "Đầu tiên chúng ta cần đăng nhập vầo Sigopt và chọn vào phần API Tokens để lấy API Token có mã như sao : JIGCTDCWFICIUSWUHDJIGFUUKSTRLWMCKRJXBDIDQILQTVFW".

import sigopt
%load_ext sigopt
%sigopt config

B4:Import thư viện.

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import datasets
import numpy
import sigopt
import time

B5:Tải tập dữ liệu "Tải tập dữ liệu sklearn và các tính năng tỷ lệ về giá trị trung bình bằng 0, phương sai đơn vị.".

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
  
B6:Bây giờ chúng ta tạo hàm chức năng cho mô hình "eval_xgboost_model khởi tạo một bộ phân loại xgboost cho mỗi lớp trong tập dữ liệu 3 lớp của chúng ta và đánh giá bộ phân loại. number_of_cross_val_folds trước khi báo cáo điểm trung bình và thời gian để khởi tạo và đào tạo các mô hình.".

#max_depth: Độ sâu tối đa cây quyết định 
#learning_rate: Thời gian học sau mỗi bước tăng cường
#min_split_loss:Giảm tổn thất tối thiểu cần thiết để thực hiện một phân vùng tiếp theo trên một nút của cây quết định.

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
    
  B7: Hàm thứ hai run_and_track_in_sigopt sử dụng các phương pháp SigOpt để ghi nhật ký và theo dõi thông tin mô hình chính bao gồm:
        Loại mô hình được sử dụng (sigopt.log_model)
        Tên của tập dữ liệu (sigopt.log_dataset)
        Các siêu tham số được sử dụng để xây dựng mô hình (sigopt.params. [PARAMETER_NAME])
        Các thuộc tính khác nhau có liên quan đến mô hình (sigopt.log_metadata)
        Số liệu đầu ra của mô hình (sigopt.log_metric).
        
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
    
B8:Với lệnh %%experiment bên dưới,chúng ta cấu hình %%experiment bằng cách đặt tên cho nó, xác định độ chính xác làm chỉ số để tối đa hóa và cuối cùng đặt không gian siêu tham số bằng cách cho SigOpt chạy các giá trị trong ranh giới đã đặt. Sau đó công cụ tối ưu hóa của SigOpt trả về các giá trị cho độ sâu tối đa từ 3 và 12 và tỷ lệ học tập là 0 và 1. Cuối cùng, xác định thời gian chúng ta sẽ đào tạo mô hình của mình. Cuối cùng chúng ta sẽ huấn luyện mô hình 4 lần, tương ứng với 4 lần chạy SigOpt.

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

B9: Chạy tối ưu
Chạy tối ưu hóa bằng cách sử dụng lệnh %% optimize. SigOpt sẽ chọn cấu hình thử nghiệm tự động và thuận tiện xuất các liên kết trong thiết bị đầu cuối tới ứng dụng chạy trên web hiện tại .
%%optimize teamdemo
run_and_track_in_sigopt()
