/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package sipres;



/*
 * FunicoApp.java
 * -Xms512m -Xmx1024m
 * -Xms32m -Xmx32m
 */

import sipres.interpreter.Evaluator;
import sipres.interpreter.ExampleException;
import sipres.interpreter.Extractor;
import sipres.interpreter.GoalException;
import sipres.interpreter.ProgramException;
import sipres.language.LexicalException;
import sipres.language.SyntacticalException;
import  java.util.Random;


/**
 *
 * @author sergio.acosta
 */
public class MyBasicNode {
    int nary;
    String type;
    MyBasicNode leftObj;
    MyBasicNode rightObj;
    Random r;
    String[] nodeOptioins = {"VARIABLE","CONSTANT","OP","S"};
    String var = "A";
    int constVar = 0;
    int treeLevel = 0;
    int treeMaxLevel = 3;
    // Types:
    /*
        ROOT,
        OP,
        S,
        VARIABLE[A,B],
        NUM_CONSTANT[0,1,2,3],
        BOOL_CONSTANT[true,false],
    */
    // Onjects null or NOT null
    public MyBasicNode(int nary, String type, MyBasicNode leftObj, MyBasicNode rightObj, int treeLevel, int treeMaxLevel) {
        this.nary = nary;
        this.type = type;
        this.leftObj = leftObj;
        this.rightObj = rightObj;
        this.treeLevel = treeLevel;
        this.treeMaxLevel = treeMaxLevel;
        this.r = new Random();
        System.out.println(this.type + " MyBasicNode Object has been created at level " + Integer.toString(this.treeLevel));
        if("VARIABLE".equals(this.type)){
            if(this.nary == 2)
            {
                int n = r.nextInt(2);
                if(n==1)
                {
                    this.var = "B";
                }
            }
            
        }
        if("CONSTANT".equals(this.type)){
            this.constVar = r.nextInt(2);   
        }
            
        
        
    }
    
    public MyBasicNode[] generate(){
        /*Create left and right nodes.
        Some times just one is required*/
        MyBasicNode[] arrayRefVar = new MyBasicNode[2];
        if("ROOT".equals(this.type))
        {
            int n = r.nextInt(4);
            MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null, (this.treeLevel + 1),this.treeMaxLevel);
            n = r.nextInt(4);
            MyBasicNode nextRight = new MyBasicNode(this.nary, nodeOptioins[n], null, null, (this.treeLevel + 1),this.treeMaxLevel);
            
            arrayRefVar[0] = nextLeft;
            arrayRefVar[1] = nextRight;
            this.leftObj = nextLeft;
            this.rightObj = nextRight;
            return arrayRefVar;   
        }
        
        if("VARIABLE".equals(this.type))
        {
           
            arrayRefVar[0] = null;
            arrayRefVar[1] = null;
            this.leftObj = null;
            this.rightObj = null;
            return arrayRefVar;
        }
        
        if("CONSTANT".equals(this.type))
        {
            
            arrayRefVar[0] = null;
            arrayRefVar[1] = null;
            this.leftObj = null;
            this.rightObj = null;
            return arrayRefVar;
        }
        
        if("S".equals(this.type))
        {
           if(this.treeLevel >= this.treeMaxLevel)
           {
                int n = r.nextInt(2);
                MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1), this.treeMaxLevel);
                arrayRefVar[0] = nextLeft;
                arrayRefVar[1] = null;
                this.leftObj = nextLeft;
                this.rightObj = null;
                return arrayRefVar; 
           }
           else
           {
               int n = r.nextInt(4);
                MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);
                arrayRefVar[0] = nextLeft;
                arrayRefVar[1] = null;
                this.leftObj = nextLeft;
                this.rightObj = null;
                return arrayRefVar; 
           }
            
        }
        
        if("OP".equals(this.type))
        {
            int options = 4;
            if(this.treeLevel >= this.treeMaxLevel)
            {
                if(this.nary == 1){
                    int n = r.nextInt(2);
                    MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);

                    arrayRefVar[0] = nextLeft;
                    arrayRefVar[1] = null;
                    this.leftObj = nextLeft;
                    this.rightObj = null;
                    return arrayRefVar;
                }
                else
                {
                    int n = r.nextInt(2);
                    MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);
                    n = r.nextInt(2);
                    MyBasicNode nextRight = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);

                    arrayRefVar[0] = nextLeft;
                    arrayRefVar[1] = nextRight;
                    this.leftObj = nextLeft;
                    this.rightObj = nextRight;
                    return arrayRefVar;
                }
            }
            else
            {
                if(this.nary == 1){
                    int n = r.nextInt(4);
                    MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);

                    arrayRefVar[0] = nextLeft;
                    arrayRefVar[1] = null;
                    this.leftObj = nextLeft;
                    this.rightObj = null;
                    return arrayRefVar;
                }
                else
                {
                    int n = r.nextInt(4);
                    MyBasicNode nextLeft = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);
                    n = r.nextInt(4);
                    MyBasicNode nextRight = new MyBasicNode(this.nary, nodeOptioins[n], null, null,(this.treeLevel + 1),this.treeMaxLevel);

                    arrayRefVar[0] = nextLeft;
                    arrayRefVar[1] = nextRight;
                    this.leftObj = nextLeft;
                    this.rightObj = nextRight;
                    return arrayRefVar;
                }
            }
            
        }
        
        
        return arrayRefVar;
    
    }
    
   public static void main(String[] args){
        int opNary = 2;
        /* Constant type 0: numeric, 1:boolean*/
        int constantType = 0;
        MyBasicNode root = new MyBasicNode(2,"ROOT", null,null, 0, 3);
        root.generate();
    }
    
}
