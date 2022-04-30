using CUDA,LinearAlgebra

# Pivot standard de matrice, en CPU
function Spivot!(M,i,j)
    M[i,:] /= M[i,j] 
    m,n = size(M)
    for k in setdiff(1:m,i)
        M[k,:] -= M[k,j]*M[i,:]
    end
    M
end

# Fonction pour trouver le pivot dans le simplexe standard en CPU
function Sfindpivot(M::Matrix{T}) where T
    entering=argmin(M[end,1:end-1])
    Mcol=M[1:end-1,end]
    leaving=argmin(map(x-> x≤0 ? typemax(T) : x, Mcol./M[1:end-1,entering]))
    return leaving,entering
end

# Fonction qui modifie le vecteur b du tableau du simplexe standard 
# pour qu'il contienne que des valeurs positives 
function standardiseProblem(lp)
    for (i,j) in enumerate(lp[1:end-1,end])
        if j<=0
            lp[i,:]*=-1
        end
    end
end

# Fonction pour savoir si notre probleme est faisable
isFeasible(b::AbstractArray{T};Δ=5) where T = all(b.≥-Δ*eps(T))

# Fonction qui crée un vecteur elementaire de taille n dans une position donnée en argument
function addArtificialVar(pos::Integer,size::Integer,::Type{T}=Float32) where T
    column=zeros(T,size)
    column[pos]=1
    column
end

# Fonction qui permets d'enlever toutes les variables artificielles
# en prenant l'union de la matrice A et b, sauf les variables artificielles
# qu'on suppose qu'ils sont tout à droite de la matrice A
removeArtificialVar(lp::AbstractMatrix,size::Integer)=lp[:,union(1:end-size-1,end)]

# Fonction qui permets de savoir si une colonne est isolée et faisable
function isisolated(v,b)
    m = length(v)
    k = iszero.(v)
    # Ici on compte le nombre de zeros dans notre colonne,
    # si il y a exactement m-1 zeros, je calcule si l'element non nul
    # du vecteur v et l'element b ont le même signe. si oui, alors la colonne est isolée et faisable
    return ((sum(k) == (m-1)) && (b[.!k][1]/v[.!k][1]≥0)), findall(.!k)[1]
end

# Fonction qui permets de savoir si la matrice A est canonique
function iscanonical(A,b)
    m, n = size(A)
    basis = zeros(Int, m)
    for i in 1:n
        # Je teste chaque colonne de ma matrice A pour voir si elle est isolée et faisable
        Ai = @view A[:, i]
        t,idx=isisolated(Ai,b)
        # Si la ieme colonne de A est isolée et faisable
        if t
            # On la rends canonique en faisant les bonnes divisions
            # et on l'ajoute dans notre vecteur de base
            b[idx]/=Ai[idx]
            A[idx,:]/=Ai[idx]
            basis[idx] = i
        end
    end
    # Nous retourne 3 éléments:
    # 1: Si la matrice A est canonique ou pas 
    # 2: Le vecteur de base qu'on a calculé
    # 3: le nombre de variables artificielles qu'on doit ajouter dans le problème pour qu'il soit canonique
    return all(!iszero, basis), basis,sum(iszero,basis)
end

mutable struct RevisedSimplex{T}
    A::CuArray{T} # La matrice A hors base initiale
    binv::CuArray{T} # La matrice Binv dans la base initiale
    b::CuArray{T} # Le tableau de droite
    originalObjective::Vector{T} # Les coûts reduits initiaux
    ci::Vector{T} # Les coûts reduits initiaux qui correspondent aux colonnes hors base
    cbi::Vector{T} # Les coûts reduits initiaux qui correspondent aux colonnes dans la base
    cb::CuArray{T} # Le vecteur de coûts reduits de base, updaté à chaque iteration du simplexe revisé
    vstar::T # La valeur objective
    xstar::Vector{T} # Le vecteur qui contient les valeurs de la solution
    bidx::Vector{Int64} # Le vecteur d'indices des colonnes de base
    access::BitVector # Un BitVector qui garde en mémoire les positions des colonnes hors base et dans la base
    # True corresponds aux colonnes de base et false corresponds aux colonnes hors base
end

# Presolve du problème, on input le problème sous la forme du simplexe standard
# Pour finalement transitionnner vers le simplexe revisé (tabulaire)
# Les calculs sont faits sur CPU 
# Une amelioration future serait d'implementer le presolve et la phase une en GPU
function RevisedSimplex(lp::AbstractMatrix{T}) where T
    # Standardisation du probleme puisqu'on suppose que le vecteur b contient que des éléments positifs
    standardiseProblem(lp)
    m,n=size(lp).-1
    # On sépare le problème linéaire en plusieurs matrices pour faciliter la transition vers le simplexe revisé
    M=@view lp[1:m,1:n]
    b=@view lp[1:m,end]
    originalProblem=lp[end,1:n]
    cv=lp[end,1:n]
    # Test pour voir si notre problème est canonique (Detection des variables de bases)

    cond,bidx,nbv=iscanonical(M,b)
    if !cond # Le problème n'est pas canonique: passons par la phase une pour trouver une base faisable
        # Trouvons les endroits ou je dois créer des variables artificielles
        zeropos=findall(iszero,bidx)
        # Creation du problème du simplexe en phase une
        phaseone=[M addArtificialVar.(zeropos,m)... b; [zeros(T,n);ones(T,nbv);zeros(T,1)]']
        for i in zeropos
            phaseone[end,:]-=phaseone[i,:]
        end

        # Pivoter pour arriver vers une base faisable 
        while !isFeasible(phaseone[end,1:n+nbv])
            leaving,entering=Sfindpivot(phaseone)
            bidx[leaving]=entering
            Spivot!(phaseone,leaving,entering)
        end

        # Si la valeur objective n'est pas égal à zero, notre problème n'est pas faisable
        (phaseone[end]>=10*eps(T)||phaseone[end]<=-10*eps(T))&&return "Non feasible"

        # Si notre base contient encore des variables artificielles
        # Ceci veut dire que notre solution de base est dégénerée
        # On voudrait alors enlever toutes les variables artificielles en faisant des pivots
        # pour faire rentrer des variables hors base (avec un coût reduit de 0) dans notre base
        nonbasic=setdiff(1:n,bidx)
        remainingArtVar=bidx[bidx.>n]
        while (!isempty(remainingArtVar))
            entering,leaving=popfirst!.([nonbasic,remainingArtVar])
            bidx[leaving]=entering
            Spivot!(phaseone,leaving,entering)
        end
        lp=removeArtificialVar(phaseone,nbv)
    end
    # On a fini avec la phase une, on a trouvé une base faisable.
    # Gardons alors en mémoire la position des colonnes de base et des colonnes hors base
    access=BitVector(zeros(n))
    access[bidx].=1
    M=@view lp[1:m,1:n]
    b=@view lp[1:m,end]
    # On remets alors les coûts reduits initiaux pour commencer la phase deux
    # Il faudrait alors annuler les coûts reduits des colonnes de base
    if !cond
        for (i,j) in enumerate(bidx)
            cv-=M[i,:]*cv[j]
            lp[end]-=b[i]*cv[j]
        end
        cv*=-1
    end
    return RevisedSimplex(CuArray(M[:,.!access]),CuArray(M[:,access]),CuArray(b),originalProblem,cv[.!access],cv[access],CuArray(cv[access]),lp[end,end],zeros(T,n),bidx,access)
end

# Pivot sous forme matricielle, calculs faits sur GPU
# binv est un CuArray, a est la colonne du vecteur entrant dans la base
# et i la position de la variable à isoler
# Tout les calculs sont faits avec de la vectorisation pour exploiter la parallelisation du GPU
function pivot!(binv,a,i)
    binv[i,:]./=a[i]
    for j in setdiff(1:size(binv,2),i)
        binv[j,:].-=binv[i,:]*a[j]
    end
end

# Fonction pour voir si notre problème est optimal, calcul fait sur GPU
# puisqu'on utilise le simplexe revisé (Tabulaire), on n'a plus accès aux coûts reduits dans un seul vecteur
# on a alors les coûts reduits qui correspondent aux colonnes de la base initiale (cb) 
# et les coûts reduits qui correspondent aux colonnes hors base (c)
isOptimal(c,cb,::Type{T}=Float32) where T=all([all(c.≥-3*eps(T)),all(cb.≥-3*eps(T))])

# Fonction pour trouver la variable entrante dans la base en GPU. 
function findEnteringVar(c,cb)
    # La fonction findmin() existe que en Julia 1.7+
    minc,argminc=findmin(c)
    mincb,argmincb=findmin(cb)
    # On retourne un boolean pour savoir si la variable entrante corresponds à une colonne en dehors de la base initiale
    # True si c'est hors base, false si c'est dans la base initiale
    mincb<minc&&return (argmincb[2],false,minc,mincb)
    return (argminc[2],true,minc,mincb)
end

# Fonction pour trouver la variable sortante de notre base en GPU.
function findExitingVar(Acol::CuArray{T},b) where T
    # On crée une fonction pos(x) pour qu'on puisse la vectoriser dans notre argmin
    # Ceci exploite alors la parallelisation du GPU.
    pos(x) = x ≤ 0 ? typemax(T) : x
    return argmin(pos.(b./Acol))
end

# Fonction pour resoudre le problème du simplexe revisé sur GPU.
function solve!(s::RevisedSimplex{T};maxIter=1000) where T
    cbinv,cbinvA=update(s)
    k=1
    while !isOptimal(cbinvA,cbinv)
        cbinvA,cbinv=update(s)
        k+=1
        k==maxIter&&break
    end
    # Calcul de la valeur objective finale
    s.xstar[s.bidx]=Array(s.binv*s.b)
    s.vstar=dot(s.xstar,s.originalObjective)
    return s.xstar,s.vstar
end

# Fonction qui permets d'iterer une fois dans le simplexe revisé sur GPU.
# Tout les produits matriciels et calculs effectués dans cette fonction sont sur GPU.
# C'est ici qu'on exploite la parallelisation du GPU le plus.
function update(s::RevisedSimplex)
    cbinv=s.cb'*s.binv
    cbinvA=cbinv*s.A-CuArray(s.ci')
    entering,t,minc,mincb=findEnteringVar(cbinvA,cbinv)
    if !isOptimal(minc,mincb)
        if t # Si la variable entrante corresponds à une colonne hors base
            Acol=s.binv*s.A[:,entering]
            leaving=findExitingVar(Acol,s.binv*s.b)
            # On convertit le CuArray Acol vers un Array sur CPU pour avoir accès aux indices de Acol
            # Et de ne pas perdre de temps en faisant des accès de memoire singuliers 
            pivot!(s.binv,Array(Acol),leaving)
            # Ici le CUDA.@allowscalar nous permets d'indexer notre CuArray sans avoir d'erreurs.
            # On peut alors acceder élément par élément notre CuArray, mais avec une perte de performance
            # Puisque l'accès est fait une seule fois par update, la perte de performance est négligeable
            CUDA.@allowscalar s.cb[leaving]=s.ci[entering]
        else 
            leaving=findExitingVar(s.binv[:,entering],s.binv*s.b)
            pivot!(s.binv,Array(s.binv[:,entering]),leaving)
            CUDA.@allowscalar s.cb[leaving]=s.cbi[entering]
        end
        # Clever way pour trouver l'indice exact dans le vrai tableau du simplexe
        # pour le garder en memoire dans bidx
        s.bidx[leaving]=findall(s.access.⊻t)[entering]
    end
    # Calcul de l'objectif
    s.vstar=dot(s.cb'*s.binv,s.b)
    return cbinvA,cbinv
end

x=[2.0f0 1 -1 0 2
   1 3 0 1 3
   1 -1 0 0 0]

xrs=RevisedSimplex(x)
solve!(xrs)

y=[1.0f0 0 3 1 1 0 0 15
2 2 0 -3 0 1 0 18
0 1 5 2 0 0 1 20
4 3 2 3 0 0 0 0]

ys=RevisedSimplex(y)

solve!(ys)


x=[-1.0f0 2 1 1 0 0 6
    1  1 0 0 1 0 24
    1 -1 1 0 0 1 9
    2  1 3 0 0 0 0]


xs=RevisedSimplex(x)
solve!(xs)