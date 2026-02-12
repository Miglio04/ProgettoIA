# Istruzioni per l'utilizzo del progetto

**Requisiti**: Avere python 3 installato sul proprio dispositivo.

1. Creare un ambiente virtuale per python

   ```bash
   python3 -m venv ambienteProgettoAi
   ```

2. Attivare l'ambiente virtuale

   ```bash
   source ambienteProgettoAi/bin/activate
   ```

3. Installare i pacchetti necessari

   ```bash
   pip install --upgrade pip
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

4. Avviare lo script desiderato;

   ```bash
   python3 nomeFile.py
   ```

5. I grafici verranno generati nella cartella `graphs` e sovrascritti ogni volta che verrà avviato lo script `models.py`.

6. Lo script `decision_tree.py` non produrrà nessun elaborato grafico.