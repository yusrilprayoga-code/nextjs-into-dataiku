import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs/promises';
import * as xlsx from 'xlsx';

export const dynamic = 'force-static';
export const revalidate = 0;

interface StructureDetails {
    field_name: string;
    structure_name: string;
    file_path: string;
    wells: string[];
    wells_count: number;
    total_records: number;
    columns: string[];
    statistics: Record<string, {
        mean?: number;
        min?: number;
        max?: number;
        count: number;
    }>;
    sample_data: Array<Record<string, unknown>>;
    data_types: Record<string, string>;
}

async function getStructureDetails(fieldName: string, structureName: string): Promise<StructureDetails> {
    const structuresDir = path.join(process.cwd(), 'data', 'structures');
    const structurePath = path.join(structuresDir, fieldName, `${structureName}.xlsx`);
    
    try {
        await fs.access(structurePath);
    } catch {
        throw new Error(`Structure file not found: ${structurePath}`);
    }

    try {
        const file = await fs.readFile(structurePath);
        const workbook = xlsx.read(file, { type: 'buffer' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const data: unknown[][] = xlsx.utils.sheet_to_json(worksheet, { header: 1 });

        if (data.length === 0) {
            throw new Error('Empty Excel file');
        }

        const headers: string[] = data[0] ? data[0].map(h => String(h)) : [];
        const wellNameIndex = headers.indexOf('Well Name');
        const wells: string[] = [];

        if (wellNameIndex !== -1) {
            for (let i = 1; i < data.length; i++) {
                const wellName = data[i][wellNameIndex];
                if (wellName && !wells.includes(String(wellName))) {
                    wells.push(String(wellName));
                }
            }
        }

        // Convert to objects for easier processing
        const records: Array<Record<string, unknown>> = data.slice(1).map((row) => {
            const obj: Record<string, unknown> = {};
            headers.forEach((header, index) => {
                obj[header] = (row as unknown[])[index];
            });
            return obj;
        });

        // Calculate statistics for numeric columns
        const statistics: Record<string, { mean?: number; min?: number; max?: number; count: number }> = {};
        const dataTypes: Record<string, string> = {};

        headers.forEach(header => {
            const values = records.map(record => record[header]).filter(val => val !== null && val !== undefined && val !== '');
            
            if (values.length === 0) {
                dataTypes[header] = 'empty';
                return;
            }

            // Check if numeric
            const numericValues = values.filter(val => !isNaN(Number(val))).map(val => Number(val));
            
            if (numericValues.length > values.length * 0.8) { // If >80% are numeric
                dataTypes[header] = 'number';
                const entry: { mean?: number; min?: number; max?: number; count: number } = { count: numericValues.length };
                if (numericValues.length > 0) {
                    entry.mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
                    entry.min = Math.min(...numericValues);
                    entry.max = Math.max(...numericValues);
                }
                statistics[header] = entry;
            } else {
                dataTypes[header] = 'string';
                statistics[header] = {
                    count: values.length
                };
            }
        });

        // Get sample data (first 10 rows)
        const sampleData = records.slice(0, 10);

        const structureDetails: StructureDetails = {
            field_name: fieldName,
            structure_name: structureName,
            file_path: structurePath,
            wells: wells.sort(),
            wells_count: wells.length,
            total_records: records.length,
            columns: headers,
            statistics,
            sample_data: sampleData as Array<Record<string, unknown>>,
            data_types: dataTypes
        };

        return structureDetails;

    } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e);
        throw new Error(`Error reading structure file: ${message}`);
    }
}

export async function GET(
    request: Request,
    { params }: { params: { fieldName: string; structureName: string } }
) {
    try {
        const data = await getStructureDetails(params.fieldName, params.structureName);
        return NextResponse.json(data);
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        return NextResponse.json({ error: message }, { status: 404 });
    }
}
