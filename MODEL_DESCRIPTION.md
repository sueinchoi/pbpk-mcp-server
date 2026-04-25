# Whole-Body PBPK Model Description

## 1. Overview

This document describes a whole-body physiologically-based pharmacokinetic (PBPK) model implemented as a Model Context Protocol (MCP) server for Claude Code. The model architecture follows the PK-Sim and Simcyp approaches, consisting of 13 tissue compartments connected through arterial and venous blood pools with a portal vein subsystem. Two distribution models are available: perfusion-limited (well-stirred organ, default) and permeability-limited (three sub-compartments per organ). The implementation includes five tissue partition coefficient (Kp) prediction methods, a complete IVIVE pipeline, Fg prediction, three hepatic clearance models, and static DDI prediction equations.

## 2. Structural Model

### 2.1 Perfusion-Limited Model

The default model treats each organ as a single well-stirred compartment (Rowland et al., 1973). The system has 16 state variables: one gut lumen depot, two blood pools (arterial and venous), and 13 tissue compartments (adipose, bone, brain, gut wall, heart, kidney, liver, lung, muscle, pancreas, skin, spleen, and rest of body).

**Circulation architecture.** The lung sits between the venous and arterial blood pools, receiving the entire cardiac output (Q_CO). All organs receive arterial blood and return venous blood. The portal vein subsystem routes venous blood from the gut wall and spleen to the liver, which also receives a direct hepatic artery supply:

$$Q_{portal} = Q_{gut} + Q_{spleen}$$
$$Q_{liver,total} = Q_{HA} + Q_{portal}$$

**Generic non-eliminating organ:**

$$\frac{dA_t}{dt} = Q_t \left( C_{art,blood} - \frac{C_t}{Kp_{b,t}} \right)$$

where $A_t$ is the amount of drug in tissue $t$, $Q_t$ is the organ blood flow, $C_{art,blood}$ is the arterial blood concentration, $C_t = A_t / V_t$ is the tissue concentration, and $Kp_{b,t} = Kp_t / R_{bp}$ is the tissue:blood partition coefficient (Rodgers and Rowland, 2007).

**Arterial blood pool:**

$$\frac{dA_{art}}{dt} = Q_{CO} \left( \frac{C_{lung}}{Kp_{b,lung}} - C_{art,blood} \right)$$

**Venous blood pool:**

$$\frac{dA_{ven}}{dt} = \sum_{i \in \text{non-portal}} Q_i \frac{C_i}{Kp_{b,i}} + Q_{liver,total} \frac{C_{liver}}{Kp_{b,liver}} - Q_{CO} \cdot C_{ven,blood} + R_{inf}$$

where the summation is over all non-portal organs (adipose, bone, brain, heart, kidney, muscle, pancreas, skin, rest of body), and $R_{inf}$ is the IV infusion rate (zero unless IV infusion route is selected).

**Lung:**

$$\frac{dA_{lung}}{dt} = Q_{CO} \left( C_{ven,blood} - \frac{C_{lung}}{Kp_{b,lung}} \right)$$

**Gut wall (with oral absorption):**

$$\frac{dA_{gut}}{dt} = Q_{gut} \left( C_{art,blood} - \frac{C_{gut}}{Kp_{b,gut}} \right) + k_a \cdot A_{lumen} \cdot F_g$$

**Gut lumen (absorption depot):**

$$\frac{dA_{lumen}}{dt} = -k_a \cdot A_{lumen}$$

The initial amount in the lumen after oral dosing is $\text{Dose} \times F_a$, where $F_a$ is the fraction absorbed from the GI tract.

**Liver (dual input, hepatic clearance):**

$$\frac{dA_{liver}}{dt} = Q_{HA} \cdot C_{art,blood} + Q_{portal} \cdot C_{portal,blood} - Q_{liver,total} \frac{C_{liver}}{Kp_{b,liver}} - CL_{int} \cdot f_{u,p} \cdot \frac{C_{liver}}{Kp_{liver}}$$

where the portal blood concentration is the flow-weighted average:

$$C_{portal,blood} = \frac{Q_{gut} \cdot C_{gut}/Kp_{b,gut} + Q_{spleen} \cdot C_{spleen}/Kp_{b,spleen}}{Q_{portal}}$$

The hepatic metabolism term uses unbound plasma concentration in the liver under well-stirred assumptions: $C_{u,liver} = f_{u,p} \cdot C_{liver} / Kp_{liver}$. For Michaelis-Menten kinetics, the elimination rate is $V_{max} \cdot C_{u,liver} / (K_m + C_{u,liver})$.

**Kidney (with renal clearance):**

$$\frac{dA_{kidney}}{dt} = Q_{kidney} \left( C_{art,blood} - \frac{C_{kidney}}{Kp_{b,kidney}} \right) - CL_{renal} \cdot \frac{C_{kidney}}{Kp_{kidney}}$$

where $CL_{renal}$ is the apparent renal clearance referenced to plasma concentration ($CL_{renal} = f_{u,p} \times GFR$ for glomerular filtration).

**Numerical solution.** The ODE system is solved using `scipy.integrate.solve_ivp` with the BDF method (backward differentiation formula), which is appropriate for stiff systems arising from the wide range of organ blood flow rates (Shampine and Reichelt, 1997). Default tolerances: $rtol = 10^{-8}$, $atol = 10^{-10}$.

### 2.2 Permeability-Limited Model

For compounds where membrane permeability is rate-limiting (typically large molecules, $MW > 700$), each organ is divided into three sub-compartments following the PK-Sim standard organ model (Willmann et al., 2003):

- **Vascular** ($V_{vas}$): blood within organ capillaries
- **Interstitial** ($V_{int}$): extracellular, extravascular space
- **Intracellular** ($V_{cell}$): intracellular space

$$\frac{dA_{vas}}{dt} = Q_t (C_{in} - C_{vas}) - PA_{endo} (f_{u,p} \cdot C_{vas}/R_{bp} - f_{u,int} \cdot C_{int})$$

$$\frac{dA_{int}}{dt} = PA_{endo} (f_{u,p} \cdot C_{vas}/R_{bp} - f_{u,int} \cdot C_{int}) - PA_{cell} (f_{u,int} \cdot C_{int} - f_{u,cell} \cdot C_{cell})$$

$$\frac{dA_{cell}}{dt} = PA_{cell} (f_{u,int} \cdot C_{int} - f_{u,cell} \cdot C_{cell})$$

where $PA_{endo}$ and $PA_{cell}$ are the permeability-surface area products for endothelial and cell membranes, respectively, and $f_{u,int}$ and $f_{u,cell}$ are the unbound fractions in interstitial and intracellular spaces. For oral dosing, absorbed drug enters the intracellular sub-compartment of the gut wall (enterocyte).

## 3. System Parameters (Physiology)

### 3.1 Organ Volumes and Blood Flows

Organ volumes are derived from ICRP Publication 89 (Valentin, 2002) as fractions of body weight, with sex-specific values for a reference adult male (73 kg) and female (60 kg). Blood flow fractions are from Williams and Leggett (1989) and ICRP 89. Cardiac output is allometrically scaled: $Q_{CO} = 15.0 \times BW^{0.74}$ L/h for males, $Q_{CO} = 13.5 \times BW^{0.74}$ L/h for females (Willmann et al., 2007).

Rest-of-body volume and blood flow are calculated as the residual after subtracting all named organs. All organ blood flow fractions sum to 1.0 as fractions of cardiac output.

### 3.2 Tissue Composition

Tissue composition data (fractional volumes of extracellular water $f_{EW}$, intracellular water $f_{IW}$, neutral lipids $f_{NL}$, neutral phospholipids $f_{NP}$, and acidic phospholipids $f_{AP}$) are from Rodgers and Rowland (2005, 2006). Intracellular pH values are set to 7.0 for most tissues (6.8 for pancreas), with extracellular and plasma pH at 7.4. Red blood cell composition ($f_{IW} = 0.603$, $f_{NL} = 0.0017$, $f_{NP} = 0.0029$, $f_{AP} = 0.0056$, $pH = 7.22$) is from Rodgers and Rowland (2005). Tissue protein fractions are from Schmitt (2008). Albumin ratios (interstitial:plasma) default to 0.5 for most tissues (0.0 for brain, 0.1 for bone).

### 3.3 Plasma Composition

Plasma water fraction 0.945, neutral lipids 0.0023, neutral phospholipids 0.0009, acidic phospholipids 0.00009 (Rodgers and Rowland, 2005). Hematocrit 0.45.

## 4. Tissue Partition Coefficient (Kp) Prediction

Five methods are implemented:

### 4.1 Rodgers and Rowland (2005, 2006)

Compounds are classified into two types based on the dominant tissue binding mechanism.

**Type 1 — Strong bases ($pK_a \geq 7$):** Electrostatic binding to acidic phospholipids (AP) dominates. The AP association constant $K_{a,AP}$ is derived from the blood:plasma ratio via RBC partitioning (Rodgers and Rowland, 2005, Eq. 6):

$$Kp_{u,RBC} = \frac{HCT - 1 + R_{bp}}{HCT \cdot f_{u,p}}$$

$$K_{a,AP} = \frac{Kp_{u,RBC} - \frac{X_{RBC}}{X_p} f_{IW,RBC} - \frac{lip_{RBC}}{X_p}}{f_{AP,RBC} \cdot \frac{X_{RBC} - 1}{X_p}}$$

where $X_{pH} = 1 + 10^{pK_a - pH}$ for bases, $lip = P \cdot f_{NL} + (0.3P + 0.7) \cdot f_{NP}$, and $P = 10^{\log P}$ (n-octanol:water).

Tissue Kp (Rodgers and Rowland, 2005, Eq. 2):

$$Kp_t = f_{u,p} \left[ f_{EW} + \frac{X_{IW}}{X_p} f_{IW} + K_{a,AP} \cdot f_{AP} \cdot \frac{X_{IW} - 1}{X_p} + \frac{lip_t}{X_p} \right]$$

**Type 2 — Acids, weak bases, neutrals, zwitterions:** Protein binding in the interstitial space dominates. $K_{a,PR}$ is derived from $f_{u,p}$ (Rodgers and Rowland, 2006, Eq. 3):

$$K_{a,PR} = \frac{1}{f_{u,p}} - 1 - \frac{lip_p}{X_p}$$

$$Kp_t = f_{u,p} \left[ f_{EW} + \frac{X_{IW}}{X_p} f_{IW} + \frac{lip_t}{X_p} + K_{a,PR} \cdot AR \cdot f_{EW} \right]$$

For adipose tissue, neutral lipid partitioning uses the vegetable oil:water partition coefficient: $\log P_{vo} = 1.115 \log P_{ow} - 1.35$ (Leo and Hansch, 1971).

### 4.2 Schmitt (2008)

Uses three lipid sub-fractions with ionization-dependent partition coefficients and an empirical protein:water partition coefficient:

$$K_{protein} = 0.163 + 0.0221 \cdot K_{n,pl}$$

where $K_{n,pl} = 10^{\log P}$ (membrane affinity). Neutral lipid partitioning is ionization-corrected: $K_{n,l} = K_{n,pl} [(1 - \alpha)/(1 + W) + \alpha]$, where $\alpha = 0.001$ is the ionized-to-neutral lipid distribution ratio and $W$ is the ionization excess. Acidic phospholipid binding is enhanced 20-fold for cationic species: $K_{a,pl} = K_{n,pl}[1/(1+W) + 20(1 - 1/(1+W))]$ for bases, and reduced ($\times 0.05$) for anions (Schmitt, 2008, Eqs. 12, 15, 19).

### 4.3 Poulin and Theil (2002)

$$Kp_t = \frac{P (f_{NL,t} + 0.3 f_{NP,t}) + (f_{W,t} + 0.7 f_{NP,t})}{P (f_{NL,p} + 0.3 f_{NP,p}) + (f_{W,p} + 0.7 f_{NP,p})} \cdot \frac{f_{u,p}}{f_{u,t}}$$

where $f_{u,t} = 1 / [1 + ((1 - f_{u,p})/f_{u,p}) \cdot AR]$. For adipose, the ionization-corrected vegetable oil:water distribution coefficient $D^*$ replaces $P$.

### 4.4 Berezhkovskiy (2004)

Corrects Poulin-Theil by incorporating $f_u$ inside the water terms:

$$Kp_t = \frac{P (f_{NL,t} + 0.3 f_{NP,t}) + 0.7 f_{NP,t} + f_{W,t}/f_{u,t}}{P (f_{NL,p} + 0.3 f_{NP,p}) + 0.7 f_{NP,p} + f_{W,p}/f_{u,p}}$$

### 4.5 PK-Sim Standard (Willmann et al., 2003)

The simplest method; does not require $pK_a$:

$$Kp_t = (f_{W,t} + K_{lipid} \cdot f_{lipid,t} + K_{protein} \cdot f_{protein,t}) \cdot f_{u,p}$$

## 5. Tissue Binding Predictions

### 5.1 Overall Tissue Unbound Fraction

At perfusion-limited equilibrium, $f_{u,tissue} = f_{u,p} / Kp$ (Rodgers and Rowland, 2007).

### 5.2 Interstitial Unbound Fraction

$$f_{u,int} = \frac{1}{1 + AR \cdot (1/f_{u,p} - 1)}$$

where $AR$ is the interstitial:plasma albumin ratio (Schmitt, 2008).

### 5.3 Intracellular Unbound Fraction

Calculated from Schmitt-style cellular partitioning:

$$f_{u,cell} = \frac{1}{f_{IW,cell} + K_{n,l} f_{NL,cell} + K_{n,pl} f_{NP,cell} + K_{a,pl} f_{AP,cell} + K_{protein} f_{prot,cell}}$$

where all fractions are normalized to cell volume.

### 5.4 Microsomal Unbound Fraction ($f_{u,inc}$)

Austin et al. (2002):

$$\log\left(\frac{1}{f_{u,inc}} - 1\right) = 0.072 (\log P)^2 + 0.067 \log P - 1.126 + \log C_{protein}$$

where $C_{protein}$ is the microsomal protein concentration (mg/mL).

## 6. In Vitro to In Vivo Extrapolation (IVIVE)

### 6.1 Microsomal CLint Scaling

$$CL_{int,\text{in vivo}} = \frac{CL_{int,\text{vitro}}}{f_{u,inc}} \times MPPGL \times LW \times \frac{60}{10^6}$$

where $CL_{int,\text{vitro}}$ is in $\mu$L/min/mg protein, $MPPGL = 45$ mg/g (Barter et al., 2007), $LW$ is liver weight in grams, and the factor converts to L/h. $f_{u,inc}$ is either measured or predicted from $\log P$ using Austin et al. (2002).

### 6.2 Hepatocyte CLint Scaling

$$CL_{int,\text{in vivo}} = \frac{CL_{int,\text{hep}}}{f_{u,hep}} \times HPGL \times LW \times \frac{60}{10^6}$$

where $HPGL = 120 \times 10^6$ cells/g (Wilson et al., 2003).

### 6.3 Recombinant CYP Scaling (ISEF)

$$CL_{int,\text{in vivo}} = \sum_j CL_{int,rCYP_j} \times ISEF_j \times A_j \times MPPGL \times LW$$

where $ISEF_j$ is the inter-system extrapolation factor (Proctor et al., 2004) and $A_j$ is the hepatic CYP abundance (pmol/mg protein) from proteomic data (Rodrigues, 1999; Rowland Yeo et al., 2004).

### 6.4 CYP Ontogeny

Enzyme maturation follows a Hill function (Johnson et al., 2006):

$$f_{mat}(age) = \frac{F_{max} \cdot age^n}{TM_{50}^n + age^n}$$

where $TM_{50}$ is the age at half-maximal expression. Pediatric MPPGL: $\log_{10}(MPPGL) = 1.407 + 0.0158 \times age_{years}$ (Barter et al., 2007).

## 7. Hepatic Clearance Models

### 7.1 Well-Stirred Model (Rowland et al., 1973)

$$CL_h = \frac{Q_h \cdot f_{u,b} \cdot CL_{int}}{Q_h + f_{u,b} \cdot CL_{int}}$$

where $f_{u,b} = f_{u,p} / R_{bp}$.

### 7.2 Parallel-Tube Model (Pang and Rowland, 1977)

$$CL_h = Q_h \left(1 - e^{-f_{u,b} \cdot CL_{int} / Q_h}\right)$$

### 7.3 Dispersion Model (Roberts and Rowland, 1986)

$$F_h = \frac{4a \cdot e^{1/(2D_N)}}{(1+a)^2 e^{a/(2D_N)} - (1-a)^2 e^{-a/(2D_N)}}$$

where $a = \sqrt{1 + 4 R_N D_N}$, $R_N = f_{u,b} \cdot CL_{int} / Q_h$, and $D_N = 0.17$ for human liver.

### 7.4 Extended Clearance Concept (Shitara et al., 2006)

For transporter-mediated clearance:

$$CL_{int,overall} = \frac{PS_{inf} (CL_{int,met} + CL_{bile})}{PS_{inf} + CL_{int,met} + CL_{bile} + PS_{eff}}$$

$$Kp_{uu,liver} = \frac{PS_{inf}}{PS_{eff} + CL_{int,met} + CL_{bile}}$$

## 8. Fg Prediction

### 8.1 Qgut Model (Yang et al., 2007)

$$F_g = \frac{Q_{gut}}{Q_{gut} + f_{u,gut} \cdot CL_{int,gut}}$$

where $Q_{gut}$ is the effective gut blood flow accounting for permeability:

$$Q_{gut} = \frac{Q_{villi} \cdot CL_{perm}}{Q_{villi} + CL_{perm}}$$

$CL_{perm} = P_{eff} \times SA_{SI}$ (permeability clearance), $P_{eff}$ is the effective human jejunal permeability, and $SA_{SI} = 2\pi r L$ is the cylindrical surface area of the small intestine ($r = 1.75$ cm, $L = 350$ cm). $Q_{villi} = 18$ L/h (Simcyp default). $f_{u,gut}$ is the enterocyte unbound fraction.

### 8.2 Caco-2 to Peff Conversion

Sun et al. (2002): $\log P_{eff} = 0.6836 \log P_{app} - 0.5579$ where $P_{eff}$ is in $10^{-4}$ cm/s and $P_{app}$ in $10^{-6}$ cm/s.

### 8.3 CAT Model for Fa (Yu and Amidon, 1999)

$$F_a = 1 - \left(1 + \frac{k_a}{k_t}\right)^{-n}$$

where $k_a = 2 P_{eff} / R$ is the absorption rate constant, $k_t = n/SITT$ is the transit rate ($SITT = 3.32$ h), and $n = 7$ compartments.

## 9. Blood:Plasma Ratio Prediction

$R_{bp}$ is predicted from RBC partitioning (Rodgers and Rowland, 2005):

$$R_{bp} = 1 - HCT + HCT \cdot Kp_{RBC}$$

where $Kp_{RBC}$ is calculated using the same Type 1/Type 2 framework applied to RBC composition.

## 10. DDI Static Prediction

### 10.1 Reversible Inhibition (FDA, 2020)

$$AUC_{ratio} = \frac{1}{f_m / (1 + [I]_{h,u}/K_i) + (1 - f_m)}$$

### 10.2 Mechanism-Based Inhibition (Fahmi et al., 2009)

$$AUC_{ratio} = \frac{1}{f_m \cdot k_{deg}/(k_{deg} + k_{obs}) + (1 - f_m)}$$

where $k_{obs} = k_{inact} [I]_u / (K_I + [I]_u)$ and $k_{deg}$ is the CYP degradation rate constant.

### 10.3 CYP Induction (FDA, 2020)

$$AUC_{ratio} = \frac{1}{f_m (1 + d \cdot E_{max} [I]_u/(EC_{50} + [I]_u)) + (1 - f_m)}$$

### 10.4 Net Effect

$$\frac{CL_{int,new}}{CL_{int,baseline}} = \underbrace{\left(1 + \frac{d E_{max} [I]}{EC_{50} + [I]}\right)}_{\text{induction}} \cdot \underbrace{\frac{k_{deg}}{k_{deg} + k_{obs}}}_{\text{MBI}} \cdot \underbrace{\frac{1}{1 + [I]/K_i}}_{\text{reversible}}$$

## 11. Software Implementation

The model is implemented as a Python MCP server using FastMCP with stdio transport. The ODE system uses SciPy's `solve_ivp` with the BDF method. All tissue composition and physiological data are embedded as Python constants. The server exposes 13 tools for Kp prediction, PBPK simulation, IVIVE, Fg prediction, hepatic clearance modeling, tissue binding, R_bp prediction, and DDI assessment.

## References

- Austin RP, Barton P, Cockroft SL, Wenlock MC, Riley RJ. The influence of nonspecific microsomal binding on apparent intrinsic clearance, and its prediction from physicochemical properties. Drug Metab Dispos. 2002;30(12):1497-1503.
- Barter ZE, Bayliss MK, Beaune PH, et al. Scaling factors for the extrapolation of in vivo metabolic drug clearance from in vitro data: reaching a consensus on values of human microsomal protein and hepatocellularity per gram of liver. Curr Drug Metab. 2007;8(1):33-45.
- Berezhkovskiy LM. Volume of distribution at steady state for a linear pharmacokinetic system with peripheral elimination. J Pharm Sci. 2004;93(6):1628-1640.
- Fahmi OA, Hurst S, Plowchalk D, et al. Comparison of different algorithms for predicting clinical drug-drug interactions, based on the use of CYP3A4 in vitro data: predictions of compounds as precipitants of interaction. Drug Metab Dispos. 2009;37(8):1658-1666.
- Johnson TN, Rostami-Hodjegan A, Tucker GT. Prediction of the clearance of eleven drugs and associated variability in neonates, infants and children. Clin Pharmacokinet. 2006;45(9):931-956.
- Leo A, Hansch C, Elkins D. Partition coefficients and their uses. Chem Rev. 1971;71(6):525-616.
- Pang KS, Rowland M. Hepatic clearance of drugs. I. Theoretical considerations of a "well-stirred" model and a "parallel tube" model. J Pharmacokinet Biopharm. 1977;5(6):625-653.
- Poulin P, Theil FP. Prediction of pharmacokinetics prior to in vivo studies. 1. Mechanism-based prediction of volume of distribution. J Pharm Sci. 2002;91(1):129-156.
- Proctor NJ, Tucker GT, Rostami-Hodjegan A. Predicting drug clearance from recombinantly expressed CYPs: intersystem extrapolation factors. Xenobiotica. 2004;34(2):151-178.
- Roberts MS, Rowland M. A dispersion model of hepatic elimination: 1. Formulation of the model and bolus considerations. J Pharmacokinet Biopharm. 1986;14(3):227-260.
- Rodgers T, Rowland M. Mechanistic approaches to volume of distribution predictions: understanding the processes. Pharm Res. 2007;24(5):918-933.
- Rodgers T, Leahy D, Rowland M. Physiologically based pharmacokinetic modeling 1: predicting the tissue distribution of moderate-to-strong bases. J Pharm Sci. 2005;94(6):1259-1276.
- Rodgers T, Rowland M. Physiologically based pharmacokinetic modelling 2: predicting the tissue distribution of acids, very weak bases, neutrals and zwitterions. J Pharm Sci. 2006;95(6):1238-1257.
- Rodrigues AD. Integrated cytochrome P450 reaction phenotyping: attempting to bridge the gap between cDNA-expressed cytochromes P450 and native human liver microsomes. Biochem Pharmacol. 1999;57(5):465-480.
- Rowland M, Benet LZ, Graham GG. Clearance concepts in pharmacokinetics. J Pharmacokinet Biopharm. 1973;1(2):123-136.
- Rowland Yeo K, Jamei M, Yang J, Tucker GT, Rostami-Hodjegan A. Physiologically based mechanistic modelling to predict complex drug-drug interactions involving simultaneous competitive and time-dependent enzyme inhibition by parent compound and its metabolite in both liver and gut. Clin Pharmacokinet. 2010;49(10):651-667.
- Schmitt W. General approach for the calculation of tissue to plasma partition coefficients. Toxicol In Vitro. 2008;22(2):457-467.
- Shampine LF, Reichelt MW. The MATLAB ODE suite. SIAM J Sci Comput. 1997;18(1):1-22.
- Shitara Y, Horie T, Sugiyama Y. Transporters as a determinant of drug clearance and tissue distribution. Eur J Pharm Sci. 2006;27(5):425-446.
- Sun D, Lennernas H, Welage LS, et al. Comparison of human duodenum and Caco-2 gene expression profiles for 12,000 gene sequences tags and correlation with permeability of 26 drugs. Pharm Res. 2002;19(10):1400-1416.
- US FDA. In Vitro Drug Interaction Studies — Cytochrome P450 Enzyme- and Transporter-Mediated Drug Interactions: Guidance for Industry. 2020.
- Valentin J. Basic anatomical and physiological data for use in radiological protection: reference values. ICRP Publication 89. Ann ICRP. 2002;32(3-4):1-277.
- Williams LR, Leggett RW. Reference values for resting blood flow to organs of man. Clin Phys Physiol Meas. 1989;10(3):187-217.
- Willmann S, Lippert J, Sevestre M, Solodenko J, Fois F, Schmitt W. PK-Sim: a physiologically based pharmacokinetic "whole-body" model. Biosilico. 2003;1(4):121-124.
- Willmann S, Schmitt W, Keldenich J, Lippert J, Dressman JB. A physiological model for the estimation of the fraction dose absorbed in humans. J Med Chem. 2004;47(16):4022-4031.
- Willmann S, Hohn K, Edginton A, et al. Development of a physiology-based whole-body population model for assessing the influence of individual variability on the pharmacokinetics of drugs. J Pharmacokinet Pharmacodyn. 2007;34(3):401-431.
- Wilson ZE, Rostami-Hodjegan A, Burn JL, et al. Inter-individual variability in levels of human microsomal protein and hepatocellularity per gram of liver. Br J Clin Pharmacol. 2003;56(4):433-440.
- Yang J, Jamei M, Yeo KR, Tucker GT, Rostami-Hodjegan A. Prediction of intestinal first-pass drug metabolism. Curr Drug Metab. 2007;8(7):676-684.
- Yu LX, Amidon GL. A compartmental absorption and transit model for estimating oral drug absorption. Int J Pharm. 1999;186(2):119-125.
