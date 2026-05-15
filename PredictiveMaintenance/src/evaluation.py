from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(y_test, y_pred, model_name):

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    print(classification_report(y_test, y_pred))

    return accuracy