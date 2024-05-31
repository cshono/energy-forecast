import gridstatus
caiso = gridstatus.CAISO()
locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]
df = caiso.get_lmp(date="today", market='DAY_AHEAD_HOURLY', locations=locations)
print(df) 