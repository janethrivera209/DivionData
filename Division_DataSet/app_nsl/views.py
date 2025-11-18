import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
from sklearn.model_selection import train_test_split
import arff   # liac-arff

# --- helpers ---
def fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def load_kdd_dataset_from_fileobj(file_obj):
    try:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw_str = raw.decode('utf-8', errors='ignore')
        else:
            raw_str = raw
        parsed = arff.loads(raw_str)
        attributes = [a[0] for a in parsed['attributes']]
        data = parsed['data']
        df = pd.DataFrame(data, columns=attributes)
        return df
    finally:
        try:
            file_obj.seek(0)
        except Exception:
            pass


# --- vista ---
def upload_file(request):
    graphs = []
    graph_titles = []
    columns = []
    rows = 0

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded = request.FILES['file']
        df = load_kdd_dataset_from_fileobj(uploaded)
        columns = df.columns.tolist()
        rows = len(df)

        # dividir en train, val, test (60% train, 20% val, 20% test)
        train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)
        val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

        # Verificar que exista la columna protocol_type
        if "protocol_type" in df.columns:
            try:
                # 1) df["protocol_type"].hist()
                plt.figure(figsize=(6,4))
                df["protocol_type"].hist()
                plt.title("Distribución de protocol_type en df")
                graphs.append(fig_to_base64())
                graph_titles.append("Distribución de protocol_type (df)")
            except Exception:
                graphs.append(None)
                graph_titles.append("Distribución de protocol_type (df)")

            try:
                # 2) train_set["protocol_type"].hist()
                plt.figure(figsize=(6,4))
                train_set["protocol_type"].hist()
                plt.title("Distribución de protocol_type en train_set")
                graphs.append(fig_to_base64())
                graph_titles.append("Distribución de protocol_type (train_set)")
            except Exception:
                graphs.append(None)
                graph_titles.append("Distribución de protocol_type (train_set)")

            try:
                # 3) val_set["protocol_type"].hist()
                plt.figure(figsize=(6,4))
                val_set["protocol_type"].hist()
                plt.title("Distribución de protocol_type en val_set")
                graphs.append(fig_to_base64())
                graph_titles.append("Distribución de protocol_type (val_set)")
            except Exception:
                graphs.append(None)
                graph_titles.append("Distribución de protocol_type (val_set)")

            try:
                # 4) test_set["protocol_type"].hist()
                plt.figure(figsize=(6,4))
                test_set["protocol_type"].hist()
                plt.title("Distribución de protocol_type en test_set")
                graphs.append(fig_to_base64())
                graph_titles.append("Distribución de protocol_type (test_set)")
            except Exception:
                graphs.append(None)
                graph_titles.append("Distribución de protocol_type (test_set)")
        else:
            graphs = [None, None, None, None]
            graph_titles = [
                "Distribución de protocol_type (df)",
                "Distribución de protocol_type (train_set)",
                "Distribución de protocol_type (val_set)",
                "Distribución de protocol_type (test_set)",
            ]

    context = {
        'graphs': zip(graphs, graph_titles),
        'columns': columns,
        'rows': rows,
    }
    return render(request, 'upload.html', context)
