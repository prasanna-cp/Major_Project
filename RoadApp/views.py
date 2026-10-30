from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph

# Global variables
global username
global X_train, X_test, y_train, y_test, X, Y, train_size
labels = ['Slight Injury', 'Serious Injury','Fatal injury']
accuracy = []
precision = []
recall = [] 
fscore = []

# Function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test, predict) * 100)
    p = (precision_score(y_test, predict, average='macro') * 100)
    r = (recall_score(y_test, predict, average='macro') * 100)
    f = (f1_score(y_test, predict, average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

# Load and preprocess dataset
dataset = pd.read_csv("Dataset/cleaned.csv")
Y = dataset['Accident_severity'].ravel()
unique, count = np.unique(Y, return_counts=True)

label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):
    name = types[j]
    if name == 'object':  # Encode string columns to numeric
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))
        label_encoder.append([columns[j], le])
dataset.fillna(dataset.mean(), inplace=True)

X = dataset.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
train_size = X_train.shape[0]

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("RF", y_test, predict)

# Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
predict = nb.predict(X_test)
calculateMetrics("NB", y_test, predict)

# SMOTE with Random Forest
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("RF+SMOTE", y_test, predict)

# SMOTE with Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
predict = nb.predict(X_test)
calculateMetrics("NB+SMOTE", y_test, predict)

# GNN Model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def build_graph(X, y, k=5):
    adj = kneighbors_graph(X, k, mode='connectivity', include_self=True)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def train_gnn(X_train, y_train, X_test, y_test, epochs=50, k=5):
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    graph = build_graph(X_all, y_all, k=k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(graph.num_node_features, 32, len(np.unique(y_all))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    graph = graph.to(device)
    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    train_mask[:len(X_train)] = True
    test_mask = ~train_mask

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.nll_loss(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = out.argmax(dim=1)
    y_true = graph.y[test_mask].cpu().numpy()
    y_pred = pred[test_mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro') * 100
    rec = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    cm = confusion_matrix(y_true, y_pred)

    return round(acc, 3), round(prec, 3), round(rec, 3), round(f1, 3), cm

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier

def PredictAction(request):
    if request.method == 'POST':
        global rf, scaler, dataset, label_encoder

        # Load dataset
        dataset = pd.read_csv("Dataset/cleaned.csv")
        Y = dataset['Accident_severity'].ravel()

        # Encode categorical features
        label_encoder = []
        columns = dataset.columns
        types = dataset.dtypes.values
        for j in range(len(types)):
            name = types[j]
            if name == 'object': 
                le = LabelEncoder()
                dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))
                label_encoder.append([columns[j], le])

        dataset.fillna(dataset.mean(), inplace=True)  # Handle missing values

        # Prepare feature matrix (X) and target vector (Y)
        # Make sure to select only the relevant features used during training
        feature_columns = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                           'Vehicle_driver_relation', 'Driving_experience', 'Lanes_or_Medians',
                           'Types_of_Junction', 'Road_surface_type', 'Light_conditions',
                           'Weather_conditions', 'Type_of_collision', 'Vehicle_movement',
                           'Pedestrian_movement', 'Cause_of_accident']  # Adjust this list
        X = dataset[feature_columns].values  # Select only the 7 features used during training
        Y = dataset['Accident_severity'].values

        # Scale the features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split (not actually needed if you're loading pre-split data)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        # Replace RandomForestClassifier with DecisionTreeClassifier
        rf = DecisionTreeClassifier()  # Using DecisionTreeClassifier
        rf.fit(X_train, y_train)

        # Get user input data from the request
        input_data = request.POST  # Assuming you pass the form data via POST
        input_values = []

        # Validate and convert each feature value to float (matching the 7 features)
        for feature in feature_columns:  # Use only the 7 features for input
            if feature in input_data:
                try:
                    value = float(input_data[feature])  # Convert to float
                    input_values.append(value)
                except ValueError:
                    input_values.append(0)  # Default to 0 if conversion fails

        # Scale the input data using the same scaler
        input_scaled = scaler.transform([input_values])

        # Predict the accident severity
        prediction = rf.predict(input_scaled)[0]

        # Map prediction to severity level
        severity_map = {
            0: '<font size="4" color="green">Slight Injury</font>',
            1: '<font size="4" color="orange">Serious Injury</font>',
            2: '<font size="4" color="red"><b>Fatal Injury</b></font>'
        }
        status = severity_map.get(prediction, '<font size="4" color="gray">Unknown</font>')

        # Create a summary of the input data
        input_summary = '<br>'.join([f"<b>{k.replace('_',' ')}:</b> {v}" for k, v in input_data.items()])

        # Output HTML
        output = f"<p><b>Prediction:</b> {status}</p>"

        # Render the result on the template
        return render(request, 'UserScreen.html', {'data': output})

def LoadDataset(request):
    if request.method == 'GET':
        global unique, count, labels, dataset

        output = '<font size="3" color="black">Dataset Loaded Successfully!</font><br/>'
        output += f'Total records found: {dataset.shape[0]}<br/>'
        output += f'Class Labels: {labels}<br/><br/>'

        if len(labels) != len(count):
            label_counts = Counter(dataset['Accident_severity'])
            labels = list(label_counts.keys())
            count = list(label_counts.values())

        height = count
        bars = labels
        y_pos = np.arange(len(bars))

        plt.figure(figsize=(4, 3))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Class Distribution")
        plt.ylabel("Count")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        context = {'data': output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def UploadDataset(request):
    if request.method == 'POST':
        if 't1' not in request.FILES:
            return render(request, 'UserScreen.html', {'data': "⚠️ No file selected!"})

        uploaded_file = request.FILES['t1']

        try:
            dataset = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        except Exception as e:
            return render(request, 'UserScreen.html', {'data': f"❌ Error: {str(e)}"})

        message = f"✅ Dataset uploaded successfully! Total records found: {dataset.shape[0]}"
        return render(request, 'UserScreen.html', {'data': message})

    return render(request, 'UploadDataset.html')

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Prasanna@593', database = 'road',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)


def BalancedData(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, train_size
        output = '<font size="3" color="black">Smote Balancing Dataset Details</font><br/>'
        output += '<font size="3" color="blue">Training Size Before Applying SMOTE = '+str(train_size)+'</font><br/>'
        output += '<font size="3" color="blue">Training Size after Applying SMOTE = '+str(X_train.shape[0])+'</font><br/><br/>'
        unique, count = np.unique(y_train)
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize = (4, 3)) 
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Balanced Dataset After Applying SMOTE Graph")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)    
def TrainML(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore, conf_matrix

        # Reset metrics lists
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()

        # 1️⃣ Random Forest
        rf = RandomForestClassifier(n_estimators=50)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        calculateMetrics("Random Forest", y_test, pred)
        conf_matrix = confusion_matrix(y_test, pred)

        # 2️⃣ Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        pred = nb.predict(X_test)
        calculateMetrics("Naive Bayes", y_test, pred)

        # 3️⃣ SMOTE + RF
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        rf_smote = RandomForestClassifier(n_estimators=50)
        rf_smote.fit(X_res, y_res)
        pred = rf_smote.predict(X_test)
        calculateMetrics("Random Forest + SMOTE", y_test, pred)

        # 4️⃣ SMOTE + Naive Bayes
        nb_smote = GaussianNB()
        nb_smote.fit(X_res, y_res)
        pred = nb_smote.predict(X_test)
        calculateMetrics("Naive Bayes + SMOTE", y_test, pred)

        # 5️⃣ Graph Neural Network
        acc, prec, rec, f1, gnn_cm = train_gnn(X_train, y_train, X_test, y_test)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        fscore.append(f1)
        conf_matrix = gnn_cm  # Optional: set confusion matrix to GNN

        # List of algorithms
        algorithms = [
            'Random Forest',
            'Naive Bayes',
            'Random Forest with SMOTE',
            'Naive Bayes with SMOTE',
            'Graph Neural Network'
        ]

        # ✅ Build metrics table
        output = '<table border=1 align=center width=100%>'
        output += '<tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F-Score</th></tr>'
        for i in range(len(algorithms)):
            output += (f'<tr>'
                       f'<td>{algorithms[i]}</td>'
                       f'<td>{accuracy[i]:.3f}</td>'
                       f'<td>{precision[i]:.3f}</td>'
                       f'<td>{recall[i]:.3f}</td>'
                       f'<td>{fscore[i]:.3f}</td>'
                       f'</tr>')
        output += "</table><br/>"

        # 🔥 Graphs: Confusion Matrix & Algorithm Comparison
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # Plot confusion matrix
        axis[0].set_title("Confusion Matrix (Latest Model)")
        sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g", ax=axis[0])

        # Plot algorithm comparison
        axis[1].set_title("Algorithm Comparison")
        df = pd.DataFrame({
            'Algorithm': algorithms,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F-Score': fscore
        })
        df.set_index('Algorithm').plot(kind='bar', ax=axis[1])

        # Save graphs as images and encode them in base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        # Pass output and image to the context for rendering
        context = {'data': output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Prasanna@593', database = 'road',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Prasanna@593', database = 'road',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Road Crash Injury  Severity Prediction"
        context= {'data':output}
        return render(request, 'Register.html', context)    


def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
import cv2
from torchvision import transforms


from django.core.files.storage import FileSystemStorage

# ================= SEVERITY CNN MODEL =================
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
class SeverityCNN(nn.Module):

    def __init__(self):
        super(SeverityCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)
# ================= LOAD TRAINED MODEL =================
severity_model = SeverityCNN()

severity_model.load_state_dict(
    torch.load("model/severity_model.pth", map_location='cpu')
)

severity_model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])
# ================= RECOMMENDATIONS =================
def get_recommendations(result):

    if result == "HIGH SEVERITY":
        return [
            "🚑 Call emergency services immediately.",
            "⚠ Provide first aid if trained.",
            "🚧 Secure accident area to avoid secondary crashes.",
            "📞 Inform nearby hospital and traffic authorities."
        ]

    else:
        return [
            "✅ Check vehicle condition before moving.",
            "🩹 Provide minor medical assistance if needed.",
            "📸 Document accident for records.",
            "🚗 Move vehicle safely from road."
        ]
def ImageSeverity(request):

    if request.method == 'GET':
        return render(request,'ImageUpload.html')

    file = request.FILES['image']

    fs = FileSystemStorage("media/images")
    name = fs.save(file.name, file)
    path = fs.path(name)

    # -------- READ IMAGE --------
    img = cv2.imread(path)
    display_img = img.copy()

    img_model = cv2.resize(img,(224,224))
    img_model = cv2.cvtColor(img_model,cv2.COLOR_BGR2RGB)

    tensor = transform(img_model).unsqueeze(0)

    # -------- PREDICT --------
    with torch.no_grad():
        pred = torch.argmax(severity_model(tensor),1).item()

    result = "HIGH SEVERITY" if pred==0 else "LOW SEVERITY"

    color = (0,0,255) if result=="HIGH SEVERITY" else (0,255,0)

    cv2.rectangle(display_img,(20,20),(450,90),color,3)

    cv2.putText(display_img,
                result,
                (30,65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3)

    # -------- SHOW WINDOW --------
    cv2.imshow("Accident Severity Result", display_img)

    while True:
        key = cv2.waitKey(1)

        # window closed
        if cv2.getWindowProperty("Accident Severity Result",
                                 cv2.WND_PROP_VISIBLE) < 1:
            break

        # ESC key
        if key == 27:
            break

    cv2.destroyAllWindows()

    # -------- SAVE OUTPUT IMAGE --------
    output_name = "output_" + name
    output_path = "media/images/" + output_name
    cv2.imwrite(output_path, display_img)

    # -------- RECOMMENDATIONS --------
    recommendations = get_recommendations(result)

    # -------- RETURN PAGE --------
    return render(request,'ImageUpload.html',{
        "result":result,
        "image":"/media/images/"+output_name,
        "recommendations":recommendations
    })
import torch
import cv2
from torchvision import transforms

# ================= ACCIDENT CNN =================
class AccidentCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*56*56,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)
# ================= LOAD ACCIDENT MODEL SAFELY =================
def load_accident_model():

    model = AccidentCNN()

    model.load_state_dict(
        torch.load("model/accident_model.pth", map_location='cpu')
    )

    model.eval()
    return model
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
import cv2
from ultralytics import YOLO
import cv2
from django.core.files.storage import FileSystemStorage
transform = transforms.ToTensor()
# Load YOLO model
from ultralytics import YOLO
import cv2
from django.core.files.storage import FileSystemStorage

# Load YOLO once
yolo_model = YOLO("yolov8n.pt")


# -------- IoU FUNCTION --------
def box_overlap(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xB < xA or yB < yA:
        return 0

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
from tensorflow.keras.models import model_from_json
import json
import numpy as np

from tensorflow.keras.models import model_from_json
import json
import numpy as np
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model


class AccidentDetectionModel:

    def __init__(self, model_path):

        self.model = load_model(model_path)

        # IMPORTANT: adjust if reversed
        self.classes = ['Accident', 'No Accident']

        print("✅ Model Loaded Successfully")

    # ---------------- PREDICT ----------------
    def predict_accident(self, img):

        preds = self.model.predict(img, verbose=0)[0]

        accident_prob = float(preds[0])   # probability of accident
        no_acc_prob = float(preds[1])

        confidence = accident_prob * 100

        # DEBUG PRINT (see terminal)
        print(f"Accident Prob: {accident_prob:.3f}")

        if accident_prob > 0.60:   # REALISTIC threshold
            label = "Accident"
        else:
            label = "No Accident"

        return label, confidence
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
import cv2
import os


# ---------------- LOAD MODEL (LOAD ONCE) ----------------
model = YOLO("runs/detect/train/weights/best.pt")

def process_video(video_path, conf_thres=0.4):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return None

    # ---------------- VIDEO FPS ----------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    delay = int(1000 / fps)

    # ⭐ SPEED CONTROL (CHANGE HERE)
    FRAME_SKIP = 2        # 1=normal, 2=2x faster, 3=3x faster

    frame_count = 0
    last_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        accident_detected = False

        # ---------------- RESIZE FOR SPEED ----------------
        small_frame = cv2.resize(frame, (640, 360))

        # ---------------- RUN YOLO ONLY SOME FRAMES ----------------
        if frame_count % FRAME_SKIP == 0:

            results = model(small_frame, conf=conf_thres, verbose=False)

            last_boxes = []

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:

                        accident_detected = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        # scale boxes back
                        scale_x = frame.shape[1] / 640
                        scale_y = frame.shape[0] / 360

                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)

                        last_boxes.append((x1, y1, x2, y2, conf))

        # ---------------- DRAW LAST DETECTIONS ----------------
        for (x1, y1, x2, y2, conf) in last_boxes:

            accident_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            label = f"Accident {conf:.2f}"
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # ---------------- ALERT ----------------
        if accident_detected:
            cv2.putText(frame, "ACCIDENT DETECTED!",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        # ---------------- SHOW VIDEO ----------------
        cv2.imshow("Accident Detection (Optimized)", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return True

# ---------------- YOUR FUNCTION ----------------
def VideoAccident(request):

    if request.method == "GET":
        return render(request, "VideoUpload.html")

    video_file = request.FILES["video"]

    fs = FileSystemStorage("media/videos")
    filename = fs.save(video_file.name, video_file)
    video_path = fs.path(filename)

    # Process and show OpenCV window
    process_video(video_path)

    return render(
        request,
        "VideoUpload.html",
        {"result": "Video Processed Successfully (Displayed in OpenCV Window)"}
    )