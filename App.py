from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import zxingcpp
import pytesseract
import uvicorn
import re
import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def limpiar_texto(texto: str) -> str:
    """Limpia caracteres nulos o no alfabéticos dejando solo letras, tildes, Ñ y espacios."""
    if not texto:
        return ""
    # Permite letras de la A-Z, a-z, acentos, Ñ/ñ y espacios
    texto_limpio = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', texto)
    return texto_limpio.strip()

# -------------------------------------------------------------
# 1. MÉTODO CÉDULA TRADICIONAL (PDF417)
# -------------------------------------------------------------
def extraer_datos_pdf417(raw_text: str):
    """Extrae datos del código de barras PDF417 """
    fecha_raw = raw_text[152:160].strip() # AAAAMMDD
    
    # Formatear la fecha a AAAA-MM-DD
    fecha_formatted = fecha_raw
    if len(fecha_raw) == 8 and fecha_raw.isdigit():
        fecha_formatted = f"{fecha_raw[:4]}-{fecha_raw[4:6]}-{fecha_raw[6:]}"

    return {
        "documento": raw_text[48:58].lstrip("0"),
        "primer_apellido": limpiar_texto(raw_text[58:81]),
        "segundo_apellido": limpiar_texto(raw_text[81:104]),
        "primer_nombre": limpiar_texto(raw_text[104:127]),
        "segundo_nombre": limpiar_texto(raw_text[127:150]),
        "sexo": raw_text[151:152].strip(),
        "fecha_nacimiento": fecha_formatted,  # AAAA-MM-DD
        "rh": raw_text[166:168].strip()
    }

# -------------------------------------------------------------
# 2. MÉTODO CÉDULA DIGITAL (MRZ OCR)
# -------------------------------------------------------------
def parsear_mrz(lineas):
    """
    Parsea las 3 líneas ICAO 9303 (MRZ) del reverso de la cédula digital.
    """
    linea1 = lineas[0]
    linea2 = lineas[1]
    linea3 = lineas[2]

    # --- A. FECHA DE NACIMIENTO Y SEXO (Línea 2: pos 0-6 YYMMDD) ---
    fecha_nac_raw = linea2[0:6] 
    sexo = linea2[7:8] if len(linea2) > 7 else ""

    fecha_nac = ""
    if len(fecha_nac_raw) == 6 and fecha_nac_raw.isdigit():
        yy = int(fecha_nac_raw[:2])
        mm = fecha_nac_raw[2:4]
        dd = fecha_nac_raw[4:6]

        anio_actual_2d = datetime.date.today().year % 100
        prefix = "20" if yy <= anio_actual_2d else "19"

        # Formato AAAA-MM-DD
        fecha_nac = f"{prefix}{yy:02d}-{mm}-{dd}"

    # --- B. NÚMERO DE DOCUMENTO (Línea 2) ---
    mrz_sub = linea2[15:] if len(linea2) > 15 else linea2
    numeros_encontrados = re.findall(r'\d{7,10}', mrz_sub)
    
    if numeros_encontrados:
        documento = numeros_encontrados[0]
    else:
        documento = re.sub(r'[^0-9]', '', mrz_sub).lstrip("0")

    # --- C. NOMBRES Y APELLIDOS (Línea 3) ---
    partes = linea3.split("<<")
    apellidos_raw = partes[0] if len(partes) > 0 else ""
    nombres_raw = partes[1] if len(partes) > 1 else ""

    apellidos = [a.replace("<", " ").strip() for a in apellidos_raw.split("<") if a]
    nombres = [n.replace("<", " ").strip() for n in nombres_raw.split("<") if n]

    primer_apellido = apellidos[0] if len(apellidos) > 0 else ""
    segundo_apellido = apellidos[1] if len(apellidos) > 1 else ""
    primer_nombre = nombres[0] if len(nombres) > 0 else ""
    segundo_nombre = " ".join(nombres[1:]) if len(nombres) > 1 else ""

    return {
        "documento": documento,
        "primer_apellido": limpiar_texto(primer_apellido),
        "segundo_apellido": limpiar_texto(segundo_apellido),
        "primer_nombre": limpiar_texto(primer_nombre),
        "segundo_nombre": limpiar_texto(segundo_nombre),
        "sexo": sexo if sexo in ["M", "F"] else "",
        "fecha_nacimiento": fecha_nac,
        "rh": ""  # El RH no forma parte del estándar ICAO MRZ
    }

def intentar_ocr_mrz(img):
    """Ejecuta Tesseract OCR buscando la franja MRZ"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    config_mrz = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    texto = pytesseract.image_to_string(gray, config=config_mrz)

    lineas_mrz = [l.strip().replace(" ", "") for l in texto.split("\n") if "<" in l and len(l.strip()) >= 15]

    if len(lineas_mrz) >= 3:
        return parsear_mrz(lineas_mrz[-3:])
    return None

# -------------------------------------------------------------
# 3. PROCESADOR UNIFICADO (PDF417 O MRZ)
# -------------------------------------------------------------
def procesar_imagen_cedula(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # --- PASO A: Wscanear código de barras PDF417 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    _, thresh_img = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variantes = [("Original", img), ("Grises", gray), ("Contraste", contrast_img), ("Binarizada", thresh_img)]

    for _, imagen_proc in variantes:
        for angulo in [0, 90, 180, 270]:
            if angulo == 90:
                img_rot = cv2.rotate(imagen_proc, cv2.ROTATE_90_CLOCKWISE)
            elif angulo == 180:
                img_rot = cv2.rotate(imagen_proc, cv2.ROTATE_180)
            elif angulo == 270:
                img_rot = cv2.rotate(imagen_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img_rot = imagen_proc

            resultados = zxingcpp.read_barcodes(img_rot, formats=[zxingcpp.BarcodeFormat.PDF417])

            if resultados:
                res = resultados[0]
                raw_text = res.bytes.decode("iso-8859-1", errors="ignore")
                if len(raw_text) >= 160:
                    datos = extraer_datos_pdf417(raw_text)
                    return {"tipo_codigo": "Lectura PDF417", "datos": datos}

    # --- PASO B: Si no hay PDF417, leer MRZ con OCR ---
    mrz_datos = intentar_ocr_mrz(img)
    if mrz_datos:
        return {"tipo_codigo": "Lectura MRZ", "datos": mrz_datos}

    return None

@app.post("/api/extraer-paciente")
async def extraer_paciente(file: UploadFile = File(...)):
    contents = await file.read()
    resultado = procesar_imagen_cedula(contents)

    if not resultado:
        raise HTTPException(
            status_code=400,
            detail="No se pudo detectar un código PDF417 ni la franja de texto MRZ."
        )

    return {
        "exito": True,
        "tipo_codigo": resultado["tipo_codigo"],
        "paciente": resultado["datos"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)