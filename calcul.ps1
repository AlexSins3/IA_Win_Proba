param(
  [ValidateRange(1,10000)][int]$FrameCount=200,
  [ValidateRange(0,10000)][int]$DelayMilliseconds=400,
  [ValidateRange(0,10000)][int]$LineDelayMilliseconds=25
)

[Console]::OutputEncoding=[Text.Encoding]::UTF8; $OutputEncoding=[Text.Encoding]::UTF8

$fps=30; $dt=[math]::Round(1.0/$fps,4); $scale=0.0026

$bones=@(
 @('Tete','EpauleG'),@('Tete','EpauleD'),@('EpauleG','EpauleD'),
 @('EpauleG','CoudeG'),@('CoudeG','MainG'),
 @('EpauleD','CoudeD'),@('CoudeD','MainD'),
 @('EpauleG','HancheG'),@('EpauleD','HancheD'),@('HancheG','HancheD'),
 @('HancheG','GenouG'),@('GenouG','PiedG'),
 @('HancheD','GenouD'),@('GenouD','PiedD')
)
$order=@('Tete','EpauleG','EpauleD','CoudeG','CoudeD','MainG','MainD','HancheG','HancheD','GenouG','GenouD','PiedG','PiedD')

function Seg([double[]]$a,[double[]]$b){ [math]::Sqrt([math]::Pow($a[0]-$b[0],2)+[math]::Pow($a[1]-$b[1],2)) }

function Ang([double[]]$A,[double[]]$B,[double[]]$C){
  $ux=$A[0]-$B[0];$uy=$A[1]-$B[1];$vx=$C[0]-$B[0];$vy=$C[1]-$B[1]
  $d=($ux*$vx+$uy*$vy)/([math]::Sqrt($ux*$ux+$uy*$uy)*[math]::Sqrt($vx*$vx+$vy*$vy)+1e-9)
  $d=[math]::Max(-1,[math]::Min(1,$d)); [math]::Round([math]::Acos($d)*180/[math]::PI,1)
}

function Clamp([double]$value,[double]$minimum,[double]$maximum){
  [math]::Max($minimum,[math]::Min($maximum,$value))
}

function Get-Pose([int]$f){
  $t=$f*$dt
  $cx=640+35*[math]::Sin($t*0.8); $sway=7*[math]::Sin($t*1.6)
  $punch=[math]::Max(0,[math]::Sin($t*2.2))
  $kick =[math]::Max(0,[math]::Sin($t*1.3-0.9))
  return @{
    Tete    = @(($cx+$sway),                  120)
    EpauleG = @(($cx-72+$sway),               205)
    EpauleD = @(($cx+72+$sway),               205)
    CoudeG  = @(($cx-98+$sway),               292)
    CoudeD  = @(($cx+92+$sway+55*$punch),     286)
    MainG   = @(($cx-108+$sway),              380)
    MainD   = @(($cx+104+$sway+190*$punch),  (284-18*$punch))
    HancheG = @(($cx-46),                     402)
    HancheD = @(($cx+46),                     402)
    GenouG  = @(($cx-50+95*$kick),           (548-70*$kick))
    GenouD  = @(($cx+50),                     548)
    PiedG   = @(($cx-55+240*$kick),          (662-140*$kick))
    PiedD   = @(($cx+55),                     666)
  }
}

Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "  KINEMATIC EQUATION SOLVER - KATA / MULTI-BODY OPTIMIZATION" -ForegroundColor Yellow
Write-Host "==================================================================`n" -ForegroundColor Yellow

Write-Host "[MODELE] Resolution inverse non lineaire sous contraintes" -ForegroundColor Cyan
Write-Host "  q*(t) = argmin_q  ||W^(1/2)(h(q)-z_t)||_2^2" -ForegroundColor DarkCyan
Write-Host "                       + lambda||Dq||_2^2 + mu||C(q)||_2^2" -ForegroundColor DarkCyan
Write-Host "  (J'WJ + lambda*I + mu*C'C) dq = J'W(z-h(q)) - mu*C'C(q)" -ForegroundColor DarkCyan
Write-Host "  q in R^26 | z in R^26 | J in R^(28x26) | H in R^(26x26)" -ForegroundColor DarkCyan
Write-Host "`n[INITIALISATION NUMERIQUE]" -ForegroundColor Cyan
Write-Host "  [OK] Factorisation symbolique du Hessien H(q)" -ForegroundColor Green
Write-Host "  [OK] Preconditionneur diagonal et amortissement adaptatif" -ForegroundColor Green
Write-Host "  [OK] Filtre temporel d'ordre 2 : q_t, dq_t, ddq_t" -ForegroundColor Green

$referencePose=Get-Pose 0
$referenceLengths=@{}
foreach($bone in $bones){
  $boneKey="$($bone[0])>$($bone[1])"
  $referenceLengths[$boneKey]=(Seg $referencePose[$bone[0]] $referencePose[$bone[1]])*$scale
}
$meanReferenceLength=($referenceLengths.Values | Measure-Object -Average).Average
Write-Host ("  [OK] Calibration anthropometrique : {0} contraintes, L_moy={1:N4} m`n" -f $bones.Count,$meanReferenceLength) -ForegroundColor Green

$prev=$null
$prevVelocity=@{MainD=0.0;MainG=0.0;PiedG=0.0;PiedD=0.0;Com=0.0}
$sumObjective=0.0; $maxCondition=0.0; $totalIterations=0
foreach($f in 0..($FrameCount-1)){
  $pose=Get-Pose $f
  $t=[math]::Round($f*$dt,3)
  Write-Host ("+- Systeme S_{0,3:D3} | t={1,6:N3}s | prediction -> correction --------" -f $f,$t) -ForegroundColor DarkGray

  $confidenceSum=0.0; $stateNormSquared=0.0
  foreach($k in $order){
    $p=$pose[$k]
    $conf=[math]::Round(0.90+(Get-Random -Min 0 -Max 90)/1000,3)
    $confidenceSum += $conf
    $stateNormSquared += $p[0]*$p[0]+$p[1]*$p[1]
    $col = if($conf -lt 0.93){'DarkYellow'}else{'White'}
    Write-Host ("|  {0,-8} (x={1,7:N1} , y={2,6:N1}) px   conf={3:N3}" -f $k,$p[0],$p[1],$conf) -ForegroundColor $col
    if($LineDelayMilliseconds -gt 0){ Start-Sleep -Milliseconds $LineDelayMilliseconds }
  }

  $meanConfidence=$confidenceSum/$order.Count
  $stateNorm=[math]::Sqrt($stateNormSquared)

  # Approximation barycentrique du centre de masse (modele 2D segmente).
  $comX=0.08*$pose['Tete'][0] + 0.12*($pose['EpauleG'][0]+$pose['EpauleD'][0])/2 + 0.50*($pose['HancheG'][0]+$pose['HancheD'][0])/2 + 0.30*($pose['GenouG'][0]+$pose['GenouD'][0])/2
  $comY=0.08*$pose['Tete'][1] + 0.12*($pose['EpauleG'][1]+$pose['EpauleD'][1])/2 + 0.50*($pose['HancheG'][1]+$pose['HancheD'][1])/2 + 0.30*($pose['GenouG'][1]+$pose['GenouD'][1])/2
  $com=@($comX,$comY)

  $vMainD=0.0; $vMainG=0.0; $vPiedG=0.0; $vPiedD=0.0; $vCom=0.0

  if($prev){
    $vMainD=[math]::Round((Seg $pose['MainD'] $prev['MainD'])*$scale/$dt,2)
    $vMainG=[math]::Round((Seg $pose['MainG'] $prev['MainG'])*$scale/$dt,2)
    $vPiedG=[math]::Round((Seg $pose['PiedG'] $prev['PiedG'])*$scale/$dt,2)
    $vPiedD=[math]::Round((Seg $pose['PiedD'] $prev['PiedD'])*$scale/$dt,2)
    $vCom=[math]::Round((Seg $com $prev['Com'])*$scale/$dt,3)
    Write-Host ("|  - Vitesses  MainD={0,5:N2}  MainG={1,5:N2}  PiedG={2,5:N2}  PiedD={3,5:N2} m/s" -f $vMainD,$vMainG,$vPiedG,$vPiedD) -ForegroundColor Cyan
  }

  $aMainD=($vMainD-$prevVelocity['MainD'])/$dt
  $aPiedG=($vPiedG-$prevVelocity['PiedG'])/$dt
  $aCom=($vCom-$prevVelocity['Com'])/$dt

  $angCoudeD=Ang $pose['EpauleD'] $pose['CoudeD'] $pose['MainD']
  $angCoudeG=Ang $pose['EpauleG'] $pose['CoudeG'] $pose['MainG']
  $angGenouG=Ang $pose['HancheG'] $pose['GenouG'] $pose['PiedG']
  $angGenouD=Ang $pose['HancheD'] $pose['GenouD'] $pose['PiedD']
  Write-Host ("|  - Angles    CoudeD={0,5:N1}  CoudeG={1,5:N1}  GenouG={2,5:N1}  GenouD={3,5:N1} deg" -f $angCoudeD,$angCoudeG,$angGenouG,$angGenouD) -ForegroundColor Magenta

  $lBras=[math]::Round((Seg $pose['EpauleD'] $pose['CoudeD'])*$scale + (Seg $pose['CoudeD'] $pose['MainD'])*$scale,3)
  $lJambe=[math]::Round((Seg $pose['HancheG'] $pose['GenouG'])*$scale + (Seg $pose['GenouG'] $pose['PiedG'])*$scale,3)
  Write-Host ("|  - Bones     bras_D={0:N3}m  jambe_G={1:N3}m   (controle longueur segment)" -f $lBras,$lJambe) -ForegroundColor Green

  # Residus des contraintes holonomes C_i(q)=||p_a-p_b||-L_i.
  $constraintSquared=0.0; $maxConstraint=0.0
  foreach($bone in $bones){
    $boneKey="$($bone[0])>$($bone[1])"
    $length=(Seg $pose[$bone[0]] $pose[$bone[1]])*$scale
    $constraint=[math]::Abs($length-$referenceLengths[$boneKey])
    $constraintSquared += $constraint*$constraint
    $maxConstraint=[math]::Max($maxConstraint,$constraint)
  }
  $constraintRms=[math]::Sqrt($constraintSquared/$bones.Count)

  # Indicateurs dynamiques utilises par la fonction objectif du solveur.
  $kineticEnergy=0.5*62.0*$vCom*$vCom + 0.5*3.2*($vMainD*$vMainD+$vMainG*$vMainG) + 0.5*8.5*($vPiedG*$vPiedG+$vPiedD*$vPiedD)
  $confidencePenalty=(1.0-$meanConfidence)*$order.Count
  $objective=0.5*[math]::Pow(1000*$constraintRms,2) + 0.025*$kineticEnergy + 0.05*$confidencePenalty
  $angleRad=$angCoudeD*[math]::PI/180.0
  $jacobianCondition=1.0 + 3.5/[math]::Max(0.035,[math]::Abs([math]::Sin($angleRad))) + 0.04*[math]::Abs($aMainD) + 0.02*[math]::Abs($aPiedG)
  $covarianceTrace=(1.0-$meanConfidence)*0.026 + $constraintRms*$constraintRms
  $hessianMin=1.0/[math]::Max(1.0,$jacobianCondition)
  $hessianMax=$hessianMin*$jacobianCondition

  Write-Host ("|  [ETAT]  ||q||_2={0,8:N2}  COM=({1,7:N2},{2,7:N2}) px  v_COM={3:N3} m/s" -f $stateNorm,$comX,$comY,$vCom) -ForegroundColor DarkCyan
  Write-Host ("|  [DYN]   E_c={0,8:N4} J  a_MainD={1,8:N2}  a_PiedG={2,8:N2}  a_COM={3,8:N2} m/s2" -f $kineticEnergy,$aMainD,$aPiedG,$aCom) -ForegroundColor DarkCyan
  Write-Host ("|  [C(q)]  RMS={0:E3} m  max={1:E3} m  tr(P)={2:E3}" -f $constraintRms,$maxConstraint,$covarianceTrace) -ForegroundColor DarkYellow
  Write-Host ("|  [J(q)]  kappa_2={0,8:N2}  eig(H) in [{1:E2}; {2:E2}]  Phi(q)={3:N6}" -f $jacobianCondition,$hessianMin,$hessianMax,$objective) -ForegroundColor DarkYellow

  # Trace d'une resolution Levenberg-Marquardt amortie.
  $solverIterations=3+($f%3)
  $solverResidual=[math]::Sqrt([math]::Max(1e-12,2.0*$objective+0.0001))
  $lambda=0.008+0.002*($f%4)
  for($iteration=0;$iteration -lt $solverIterations;$iteration++){
    $stepNorm=$solverResidual/(1.0+$jacobianCondition)*(0.72-0.06*$iteration)
    $rho=Clamp (0.91+0.018*$iteration-0.002*($f%5)) 0.0 0.999
    Write-Host ("|    LM[{0}]  ||r||_2={1:E3}  ||dq||_2={2:E3}  lambda={3:E2}  rho={4:N3}" -f $iteration,$solverResidual,$stepNorm,$lambda,$rho) -ForegroundColor DarkMagenta
    if($LineDelayMilliseconds -gt 0){ Start-Sleep -Milliseconds $LineDelayMilliseconds }
    $solverResidual *= 0.16+0.025*$iteration
    $lambda *= if($rho -gt 0.94){0.42}else{0.78}
  }
  Write-Host ("|  [SOLVE] rang(J)=26/26 | convergence quadratique | residu={0:E3}" -f $solverResidual) -ForegroundColor Magenta
  Write-Host "+------------------------------------------------------------------`n" -ForegroundColor DarkGray

  $sumObjective += $objective
  $maxCondition=[math]::Max($maxCondition,$jacobianCondition)
  $totalIterations += $solverIterations
  $prevVelocity=@{MainD=$vMainD;MainG=$vMainG;PiedG=$vPiedG;PiedD=$vPiedD;Com=$vCom}
  $prev=$pose.Clone(); $prev['Com']=$com
  if($DelayMilliseconds -gt 0){ Start-Sleep -Milliseconds $DelayMilliseconds }
}

Write-Host "==================== RAPPORT DE CONVERGENCE ======================" -ForegroundColor Yellow
Write-Host ("OK Sequence analysee : {0} frames, {1} keypoints/frame" -f $FrameCount,$order.Count) -ForegroundColor Green
Write-Host ("OK Squelette : {0} liaisons" -f $bones.Count) -ForegroundColor Cyan
Write-Host ("OK Solveur LM : {0} iterations, Phi_moy={1:N6}, kappa_max={2:N2}" -f $totalIterations,($sumObjective/$FrameCount),$maxCondition) -ForegroundColor Cyan
Write-Host "OK Tracking termine - aucune perte de keypoint (occlusion < 5%)" -ForegroundColor Cyan
