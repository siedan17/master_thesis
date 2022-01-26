---
title: 'Masterarbeit Siemmeister: Ausgangsmodelle'
author: "Markus Schweighart"
date: "16 4 2021"
output: html_document
---

## Packages
```{r}
library(tidyverse)
library(randomForest)
library(pscl)
library(sjPlot)
```

## Datensatz laden
```{r}
da242_sj <- read.csv("da242_BWL_PAD_JUD_sj.csv")

library(readr) ## habe ich hier hinzugefuegt!
da242_BWL_PAD_JUD_sj <- read_csv("programming/masterarbeit/code/plain_version/da242_BWL_PAD_JUD_sj.csv")

```

## Trennung: Studienjahr 1 / ab Studienjahr 2
```{r}
sj1 <- da242_sj %>% filter(studiendauer_semester <= 2)
sjab2 <- da242_sj %>% filter(studiendauer_semester > 2)
``` 

## Trennung: Trainings- (15/16 - 18/19) / Testdaten (19/20)
```{r}
training_sj1 <- sj1 %>% filter(SEMESTER_NAME %in%  c('15W', '16S', '16W', '17S', '17W', '18S', '18W', '19S'))
test_sj1 <- sj1 %>% filter(SEMESTER_NAME %in%  c('19W', '20S'))
 
training_sjab2 <- sjab2 %>% filter(SEMESTER_NAME %in%  c('15W', '16S', '16W', '17S', '17W', '18S', '18W', '19S'))
test_sjab2 <- sjab2 %>% filter(SEMESTER_NAME %in%  c('19W', '20S'))
```






################# Ausgangsmodelle ######################
########################################################




## Lineares Regressionsmodell
```{r}
########## 1. Studienjahr ############
######################################
sj1_Model_lm <- training_sj1 %>% lm(ECTS_betr_Studienjahr ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + STUDIENRICHTUNG, data =  .)

# Modellüberblick
summary(sj1_Model_lm)
tab_model(sj1_Model_lm, show.std = T, show.ci = F)

# Prediction für 19/20
test_sj1$prediction_lm <- predict(sj1_Model_lm, newdata = test_sj1, se.fit = F, type = "response")
test_sj1$prediction_aktiv_lm <- ifelse(test_sj1$prediction_lm >= 16, 1,0) # nur ECTS!


####### ab 2. Studienjahr ############
######################################
sjab2_Model_lm <- training_sjab2 %>% lm(ECTS_betr_Studienjahr ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + studiendauer_semester +  bisherige_ects_sem + STUDIENRICHTUNG, data =  .)

# Modellüberblick
summary(sjab2_Model_lm)
tab_model(sjab2_Model_lm, show.std = T, show.ci = F)
# Multikollinearitätscheck:
vif(sjab2_Model_lm)

# Prediction für 19/20 und Anteil richtiger Predictions
test_sjab2$prediction_lm <- predict(sjab2_Model_lm, newdata = test_sjab2, se.fit = F, type = "response")
test_sjab2$prediction_aktiv_lm <- ifelse(test_sjab2$prediction_lm >= 16, 1,0)
test_sjab2 %>% mutate(richtig_pred = case_when(prediction_aktiv_lm == beobachtet_aktiv_dummy ~ 1, TRUE ~ 0)) %>% group_by(Studienjahr) %>%
  summarise(Anteil_beob_pa = mean(beobachtet_aktiv_dummy, na.rm = T),
            Anteil_pred_pa = mean(prediction_aktiv_lm, na.rm = T),
            Anteil_richtig_pred = mean(richtig_pred, na.rm = T),
            Anteil_richtig_pred_pa = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 1, na.rm = T) / sum(beobachtet_aktiv_dummy == 1),
            Anteil_richtig_pred_nicht_na = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 0, na.rm = T) / sum(beobachtet_aktiv_dummy == 0))

# QS
# test <- test_sj1 %>% as.data.frame() %>% mutate(richtig_pred = case_when(beobachtet_aktiv_dummy == prediction_aktiv_lm ~ 1, T ~ 0)) %>% select(MATRIKELNUMMER, prediction_lm, prediction_aktiv_lm, beobachtet_aktiv_dummy, richtig_pred) %>% slice(sample(1:length(test_sj1), 100))
```





## Logistischen Regressionsmodell
```{r}
########## 1. Studienjahr ############
######################################
sj1_Model_binlog <- training_sj1 %>% glm(beobachtet_aktiv_dummy ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + STUDIENRICHTUNG, data =  ., family = binomial())

# Modelüberblick
summary(sj1_Model_binlog)
tab_model(sj1_Model_binlog, show.std = T, show.ci = F)

# Prediction für 19/20
test_sj1$prediction_binlog <- predict(sj1_Model_binlog, newdata = test_sj1, se.fit = F, type = "response")
test_sj1 <- test_sj1 %>% mutate(prediction_aktiv_binlog = case_when(prediction_binlog >= .5 ~ 1, T ~ 0))



####### ab 2. Studienjahr ############
######################################
sjab2_Model_binlog <- training_sjab2 %>% glm(beobachtet_aktiv_dummy ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + studiendauer_semester +  bisherige_ects_sem + STUDIENRICHTUNG, data =  ., family = binomial())

# Modelüberblick
summary(sjab2_Model_binlog)
tab_model(sjab2_Model_binlog, show.std = T, show.ci = F)

# Prediction für 19/20 und Anteil richtiger Predictions
test_sjab2$prediction_binlog <- predict(sjab2_Model_binlog, newdata = test_sjab2, se.fit = F, type = "response")
test_sjab2 <- test_sjab2 %>% mutate(prediction_aktiv_binlog = case_when(prediction_binlog >= .5 ~ 1, T ~ 0))

test_sjab2 %>% mutate(richtig_pred = case_when(prediction_aktiv_binlog == beobachtet_aktiv_dummy ~ 1, TRUE ~ 0)) %>% group_by(Studienjahr) %>%
  summarise(Anteil_beob_pa = mean(beobachtet_aktiv_dummy, na.rm = T),
            Anteil_pred_pa = mean(prediction_aktiv_binlog, na.rm = T),
            Anteil_richtig_pred = mean(richtig_pred, na.rm = T),
            Anteil_richtig_pred_pa = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 1, na.rm = T) / sum(beobachtet_aktiv_dummy == 1),
            Anteil_richtig_pred_nicht_na = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 0, na.rm = T) / sum(beobachtet_aktiv_dummy == 0))
```











## Random Forest Modell
```{r}
########## 1. Studienjahr ############
######################################
sj1_Model_forest <- training_sj1 %>% drop_na(jahre_seit_18) %>% randomForest(ECTS_betr_Studienjahr ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + STUDIENRICHTUNG, data =  .)

# Prediction für 19/20
test_sj1$prediction_forest <- predict(sj1_Model_forest, newdata = test_sj1, se.fit = F, type = "response")
test_sj1 <- test_sj1 %>% mutate(prediction_aktiv_forest = case_when(prediction_forest >= 16 ~ 1, T ~ 0))

test_sj1 %>% mutate(richtig_pred = case_when(prediction_aktiv_forest == beobachtet_aktiv_dummy ~ 1, TRUE ~ 0)) %>% group_by(Studienjahr) %>%
  summarise(Anteil_beob_pa = mean(beobachtet_aktiv_dummy, na.rm = T),
            Anteil_pred_pa = mean(prediction_aktiv_forest, na.rm = T),
            Anteil_richtig_pred = mean(richtig_pred, na.rm = T),
            Anteil_richtig_pred_pa = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 1, na.rm = T) / sum(beobachtet_aktiv_dummy == 1),
            Anteil_richtig_pred_nicht_na = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 0, na.rm = T) / sum(beobachtet_aktiv_dummy == 0))

####### ab 2. Studienjahr ############
######################################
sjab2_Model_forest <- training_sjab2 %>% drop_na(jahre_seit_18) %>% randomForest(ECTS_betr_Studienjahr ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + studiendauer_semester +  bisherige_ects_sem + STUDIENRICHTUNG, data =  .)

# Prediction für 19/20 und Anteil richtiger Predictions
test_sjab2$prediction_forest <- predict(sjab2_Model_forest, newdata = test_sjab2, se.fit = F, type = "response")
test_sjab2 <- test_sjab2 %>% mutate(prediction_aktiv_forest = case_when(prediction_forest >= 16 ~ 1, T ~ 0))

test_sjab2 %>% mutate(richtig_pred = case_when(prediction_aktiv_forest == beobachtet_aktiv_dummy ~ 1, TRUE ~ 0)) %>% group_by(Studienjahr) %>%
  summarise(Anteil_beob_pa = mean(beobachtet_aktiv_dummy, na.rm = T),
            Anteil_pred_pa = mean(prediction_aktiv_forest, na.rm = T),
            Anteil_richtig_pred = mean(richtig_pred, na.rm = T),
            Anteil_richtig_pred_pa = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 1, na.rm = T) / sum(beobachtet_aktiv_dummy == 1),
            Anteil_richtig_pred_nicht_na = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 0, na.rm = T) / sum(beobachtet_aktiv_dummy == 0))
```





## Zero Inflation Modell
```{r}
########## 1. Studienjahr ############
######################################
sj1_Model_zinf <- training_sj1 %>% drop_na(jahre_seit_18) %>% zeroinfl(as.integer(ECTS_betr_Studienjahr) ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + STUDIENRICHTUNG, data =  .)

# Prediction für 19/20
test_sj1$prediction_zinf <- predict(sj1_Model_zinf, newdata = test_sj1, se.fit = F, type = "response")
test_sj1 <- test_sj1 %>% mutate(prediction_aktiv_zinf = case_when(prediction_zinf >= 16 ~ 1, T ~ 0))



####### ab 2. Studienjahr ############
######################################
sjab2_Model_zinf <- training_sjab2 %>% drop_na(jahre_seit_18) %>% zeroinfl(as.integer(ECTS_betr_Studienjahr) ~ dummy_weiblich  + jahre_seit_18 +  zulassung_tag_num + sonstiges_at_dummy + deutschland_dummy + sonstiges_ausland_dummy + Realgymnasium_Dummy  + BHS_Dummy  + Sonstige_Vorbildung_Dummy + studienbeginn_sommersemester  + mehrere_studien + studiendauer_semester +  bisherige_ects_sem + STUDIENRICHTUNG, data =  .)

# Prediction für 19/20 und Anteil richtiger Predictions
test_sjab2$prediction_zinf <- predict(sjab2_Model_zinf, newdata = test_sjab2, se.fit = F, type = "response")
test_sjab2 <- test_sjab2 %>% mutate(prediction_aktiv_zinf = case_when(prediction_zinf >= 16 ~ 1, T ~ 0))

test_sjab2 %>% mutate(richtig_pred = case_when(prediction_aktiv_zinf == beobachtet_aktiv_dummy ~ 1, TRUE ~ 0)) %>% group_by(Studienjahr) %>%
  summarise(Anteil_beob_pa = mean(beobachtet_aktiv_dummy, na.rm = T),
            Anteil_pred_pa = mean(prediction_aktiv_zinf, na.rm = T),
            Anteil_richtig_pred = mean(richtig_pred, na.rm = T),
            Anteil_richtig_pred_pa = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 1, na.rm = T) / sum(beobachtet_aktiv_dummy == 1),
            Anteil_richtig_pred_nicht_na = sum(richtig_pred == 1 & beobachtet_aktiv_dummy == 0, na.rm = T) / sum(beobachtet_aktiv_dummy == 0))

```