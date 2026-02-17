import subprocess
import json
import os
import ast

def compterLigne(fichier : str) -> tuple:
    """compte le nombre de ligne et le nombre de ligne commentées (prennant en comtpe la docString)
    Args:
        fichier (str): le path du fichier
    Returns:
        tuple: nombre de ligne total, nombre de ligne commentées"""
    try:
        result = subprocess.run([
            "cloc", fichier, "--json"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print("Erreur lors de l'exécution de cloc. Assurez-vous qu'il est installé.")
            return
        data = json.loads(result.stdout)
        if "Python" in data:
            ligneTotal = data["Python"]["code"] + data["Python"]["comment"]
            ligneCommentee = data["Python"]["comment"]
            return ligneTotal, ligneCommentee
        else:
            print("Aucune donnée trouvée pour Python. Assurez-vous que le fichier est valide.")
    except FileNotFoundError:
        print("Fichier introuvable. Veuillez vérifier le chemin.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")


def compterVaraiblesNonDeclarees(pathFichier : str) -> int:
    """compte le nombre de variables non déclarées
    Args:
        pathFichier (str): le path du fichier
    Returns:
        int: le nombre de variables non déclarées"""
    try:
        result = subprocess.run(["pylint", "--errors-only", pathFichier], capture_output=True, text=True)
        errors = result.stdout
        variablesNonDeclarrees = []
        for line in errors.split('\n'):
            if "E0602" in line:
                variablesNonDeclarrees.append(line.strip())
        
        if variablesNonDeclarrees:
            return len(variablesNonDeclarrees)
        else:
            return 0
    except Exception as e:
        print(f"Erreur lors de l'exécution de pylint: {e}")

def compteFonction30Lignes(pathFichier : str) -> int:
    """Compte le nombre de fontions de plus de 30 lignes sans prendre en compte la docString et les commentaires
    Args:
        pathFichier (str): le path du fichier
    Returns:
        int: le nombre de fonction de plus de 30 lignes"""
    with open(pathFichier, 'r', encoding='utf-8') as file:
        code = file.read()
    def retireDocString(code):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)

        return ast.unparse(tree)
    cleaned_code = retireDocString(code)
    def compteLigne(code):
        lines = code.split('\n')
        count = 0
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):
                count += 1
        return count
    tree = ast.parse(cleaned_code)
    long_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
            function_code = "\n".join(code.splitlines()[start_line - 1:end_line])
            code_lines = compteLigne(function_code)
            if code_lines > 30:
                long_functions.append(node.name)
    return len(long_functions)


def analyseDossier(dossier : str ) -> None:
    """applique les test implémenter plus haut sur tous les fichier du dossier donné
    Args:
        dossier (str): le path du dossier à analyser"""
    if not os.path.isdir(dossier):
        raise FileNotFoundError(f"Le dossier {dossier} n'existe pas.")
    for root, dirs, files in os.walk(dossier):
        for file in files:
            if file.endswith(".py"):
                ligneTotal, ligneCommentee = compterLigne(os.path.join(root, file))
                nbVarNonDecl = compterVaraiblesNonDeclarees(os.path.join(root, file))
                nbFonction30Lignes = compteFonction30Lignes(os.path.join(root, file))
                print(f"Pour le fichier {os.path.join(root, file)} : poucentage commentaire ({ligneCommentee/ligneTotal*100:.2f}%), variables non déclarées ({nbVarNonDecl}), fonction plus de 30 lignes ({nbFonction30Lignes}) ")

if __name__ == "__main__":
    analyseDossier("./")