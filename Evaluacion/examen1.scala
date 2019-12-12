var input0 = List(10,5,20,20,4,5,2,25,1)
var input1 = List(3,4,21,36,10,28,35,5,24,42)
//Define el ciclo
def breakingRecords (nums:List[Int]) : Unit =
{    
    //busca el mejor record
    var max, min = nums(0)
    //buscar el peor record
    var lowest, highest = 0
    //Inicia el recorrido
    for (i <- nums)
    {
        //Busca el lugar donde el siguiente lugar es el mejor record
        if (i>max)
        {max = i
            highest = highest + 1
            }
        //Busca el lugar donde el siguiente lugar es el peor record
        if (i<min)
        {min = i
            lowest = lowest +1
            }
    }
    //imprime el mejor record descues el peor record
    println (highest,lowest)
}
//Imprimir el input0
breakingRecords(input0)
//Imprimir el input1
breakingRecords(input1)