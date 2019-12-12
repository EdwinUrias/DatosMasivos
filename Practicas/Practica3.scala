//ALGORITMO 1
def fib1(n: Int): Int = 
{
    if (n<2) 
    {
        return (n)
    }
    else
    {
        return ((fib1(n-1))+(fib1(n-2)))
    }
}

//ALGORITMO 2
def fib2(n: Int): Double = 
{
    var p = 0.0
    var j = 0.0
    if (n<2)
    {
        return (n)
    }
    else
    {
        p = ((1+math.sqrt(5))/2)
        j = (((math.pow(p, n))-(math.pow((1-p),n)))/(math.sqrt(5)))
    }
    return j
}

//ALGORITMO 3
def fib3(n: Int): Double = 
{
    var a=0.0
    var b=1.0
    var c=0.0
    for (k <- 0 to n-1)
    {
        c = b + a
        a = b
        b = c
    }
    return a
}

//ALGORITMO 4
def fib4(n: Int): Double = 
{
    var a=0.0
    var b=1.0
    for (k <- 0 to n-1)
    {
        b = b + a
        a = b - a
    }
    return a
} 
//ALGORITMO 5
def fib5 (n: Int) : Double =
{
    if (n <2)
    {
        return n
    }
    else
    {
        vector= Array.range(0,n+1)
        vector(0) = 0
        vector(1) = 1
        for (k <- 2 to n)
        {
            vector(k) = vector(k-1) + vector (k-2)
        }
        return vector(n)
    }
}

//ALGORITMO 6
def fib6 (n : Int) : Double =
{
    if (n <= 0)
    {
        return 0
    }
    var i = n-1
    var auxOne = 0.0
    var auxTwo = 1.0
    var ab = Array(auxTwo,auxOne)
    var cd = Array(auxOne,auxTwo)
    while (i>0)
    {
        if (i % 2 != 0)
        {
            auxOne = cd(1) * ab(1) + cd(0) * ab(0)
            auxTwo = cd(1) * (ab(1)+ab(0)) + cd(0)*ab(1)
            ab(0) = auxOne
            ab(1) = auxTwo 
        } 
        auxOne = (math.pow(cd(0),2)) + (math.pow(cd(1),2))
        auxTwo = cd(1)* (2*cd(0) + cd(1))
        cd(0) = auxOne
        cd(1) = auxTwo
        i = i/2
    }
    return (ab(0) + ab(1))
}