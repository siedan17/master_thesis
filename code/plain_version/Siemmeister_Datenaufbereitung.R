---
title: 'Masterarbeit Siemmeister: Datenaufbereitung'
author: "Markus Schweighart"
date: "16 4 2021"
output: html_document
---

```{r}
library(tidyverse)
library(lubridate)
```



## da242: Reduktion auf 15W-20S + zusätzliche Variablen 
```{r}
da242 <- readRDS(file = "Y://Studierendendaten/00_Studienverlaufsanalysen_abJuli19/USTAT Ideen und Auswertungen/Datensatzaufbereitung da242 x USTAT  x UHSTAT/Output - angereicherte Files (rds, csv)/da242_ustat_uhstat.rds")




###### Reduktion auf 15W-20S #############################################
#########################################################################

# Filter (15W-19S) und Studienjahr
da242 <- da242 %>% filter(SEMESTER_NAME %in% c("15W", "16S", "16W", "17S", "17W", "18S", "18W", "19S", "19W", "20S")) %>% 
  mutate(Studienjahr = case_when(SEMESTER_NAME %in% c("15W", "16S") ~ "15_16",
                                 SEMESTER_NAME %in% c("16W", "17S") ~ "16_17",
                                 SEMESTER_NAME %in% c("17W", "18S") ~ "17_18",
                                 SEMESTER_NAME %in% c("18W", "19S") ~ "18_19",
                                 SEMESTER_NAME %in% c("19W", "20S") ~ "19_20"))

###### Variablenerstellung. Feature Selection & Engineering #############
#########################################################################

## Kooperations-Prüfungsleistungen: Die Summe aus Anrechnungen und Prüfungsleistungen kommt als ECTS-Variable zum Einsatz:
da242$SUMECTS_CREDIT_POS <- ifelse(da242$kooperationsstudium_dummy==1, da242$SUM_ECTS_POS_GESAMT, da242$SUMECTS_CREDIT_POS)
da242$SUMSTD_POS <- ifelse(da242$kooperationsstudium_dummy==1, da242$SUM_SWS_POS_GESAMT, da242$SUMSTD_POS)

## Zulassungsdatum numerisch [anstelle von Nachfrist-Dummy einsetzbar]
da242$zulassung_tag_num <- ifelse(da242$studienbeginn_sommersemester==1, yday(da242$ZULASSUNGSDATUM) - 36, yday(da242$ZULASSUNGSDATUM) - 274)

## Zusammenfassung Gymnasium und Realgymnasium zu Schultyp "AHS" ##
da242$AHS_Dummy <- ifelse((da242$Gymnasium_Dummy + da242$Realgymnasium_Dummy) != 0, 1, 0 )

## Erstellung Dummy für Zulassung in Nachfrist ##
da242$Nachfrist_Dummy <- ifelse(da242$zulassung_tag_num > 0, 1, 0)
# NAs als "0" deklarieren
da242$Nachfrist_Dummy <- ifelse(da242$Nachfrist_Dummy==0 | is.na(da242$Nachfrist_Dummy),0,1)

# Matrikelnummer-Studidf
da242 <- da242 %>% mutate(MatrklStudidf = paste0(MATRIKELNUMMER, STUDIDF))
```




## Betrachtung pro Studienjahr
```{r}
# ECTS/SWS/Prüfungsaktivität pro Studienjahr (sj)
da242_sj <- da242 %>% group_by(MatrklStudidf, Studienjahr) %>% 
  mutate(ECTS_betr_Studienjahr = sum(SUMECTS_CREDIT_POS), SWS_betr_Studienjahr = sum(SUMSTD_POS), ECTS_betr_Studienjahr_num_rd = round(ECTS_betr_Studienjahr)) %>% arrange(desc(STUDIENDAUER)) %>% slice(1) %>%
  mutate(beobachtet_aktiv_dummy = case_when(ECTS_betr_Studienjahr >= 16 | SWS_betr_Studienjahr >= 8 | STUDIENSTATUS_KURZ == "X" ~ 1, TRUE ~ 0))

## ECTS pro vorangegangenem Semester ##
da242_sj <- da242_sj %>% mutate(studiendauer_semester = STUDIENDAUER / 182.5,
                                bisherige_ects = kum_ects_pos - SUMECTS_CREDIT_POS, # is there a mistake? SUMECTS_CREDIT_POS is only the ECTS of the latest semester?! Because above there is sum.
                                bisherige_ects_sem = bisherige_ects / studiendauer_semester) # is there a mistake? bisherige_ects is before this year and studiendauer_semester is with this year.
```




## Auswahl: Bachelor/Diplom & Studienrichtungen
```{r}
da242_sj <- da242_sj %>% filter(STUDIENART_KBEZ %in% c("Bachelor", "Diplom") & STUDIENRICHTUNG %in% c("Rechtswissenschaften", "Padagogik", "Betriebswirtschaft"))

write.csv(da242_sj, "da242_BWL_PAD_JUD_sj.csv")
```


